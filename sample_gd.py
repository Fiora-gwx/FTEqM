# 本源代码的许可证位于根目录的 LICENSE 文件中

"""
使用 PyTorch DDP 的 EqM 最小采样脚本
用于从训练好的 EqM 模型生成图像样本
"""
import math
import torch
# 下面的第一个标志在测试时为 False,但设为 True 会使 A100 训练快得多:
torch.backends.cuda.matmul.allow_tf32 = True  # 允许在矩阵乘法中使用 TensorFloat-32
torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TensorFloat-32
import torch.distributed as dist  # 分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm
from models import EqM_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL  # 变分自编码器
from train_utils import parse_transport_args
import wandb_utils
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    从包含 .png 样本的文件夹构建单个 .npz 文件
    
    .npz 格式用于存储大量图像,便于后续进行 FID (Fréchet Inception Distance) 评估
    
    参数:
        sample_dir: 样本图像所在文件夹路径
        num: 要打包的图像数量,默认 50,000 张
        
    返回:
        npz_path: 生成的 .npz 文件路径
    """
    samples = []
    # 遍历所有样本图像
    for i in tqdm(range(num), desc="从样本构建 .npz 文件"):
        # 读取 PNG 图像(命名格式: 000000.png, 000001.png, ...)
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        # 转换为 numpy 数组,数据类型为 uint8 (0-255)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    # 将所有样本堆叠成一个大数组
    samples = np.stack(samples)
    # 验证形状: (数量, 高度, 宽度, 3通道RGB)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    
    # 保存为 .npz 文件
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"已保存 .npz 文件到 {npz_path} [shape={samples.shape}].")
    return npz_path


def requires_grad(model, flag=True):
    """
    为模型中的所有参数设置 requires_grad 标志
    
    参数:
        model: 模型对象
        flag: 是否需要梯度,True 表示需要,False 表示冻结参数
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    结束 DDP (分布式数据并行) 训练
    清理分布式进程组
    """
    dist.destroy_process_group()


def main(args):
    """
    使用训练好的 EqM 模型生成图像样本
    
    主要步骤:
    1. 初始化分布式环境和模型
    2. 加载训练好的检查点
    3. 使用迭代采样生成图像
    4. 将生成的图像保存为 PNG 和 NPZ 格式
    
    参数:
        args: 命令行参数对象,包含所有采样配置
    """
    # 验证 CUDA 可用性
    assert torch.cuda.is_available(), "采样当前至少需要一个 GPU。"
    n_gpus = torch.cuda.device_count()
    
    # 如果启用能量基模型(EBM),禁用 Flash Attention 等优化
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    # 设置 DDP (分布式数据并行):
    dist.init_process_group("nccl")  # 使用 NCCL 后端进行 GPU 通信
    assert args.global_batch_size % dist.get_world_size() == 0, f"批次大小必须能被进程数整除。"
    
    # 获取当前进程的 rank 和设备编号
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"找到 {n_gpus} 个 GPU,尝试使用设备索引 {device}")
    
    # 设置随机种子(每个进程使用不同的种子)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"启动 rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # 计算每个进程的批次大小
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    
    # 创建模型:
    assert args.image_size % 8 == 0, "图像尺寸必须能被 8 整除(用于 VAE 编码器)。"
    latent_size = args.image_size // 8  # 潜在空间尺寸(VAE 压缩后的尺寸)
    
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,  # 是否启用无条件生成
        ebm=args.ebm  # 能量基模型类型
    ).to(device)

    # 注意:参数初始化在 EqM 构造函数内完成
    ema = deepcopy(model).to(device)  # 创建模型的 EMA 副本用于采样

    # 加载训练好的检查点
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            # 完整检查点格式(包含 model 和 ema)
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
        else:
            # 仅权重格式(通常是 EMA 权重)
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)

        ema = ema.to(device)
        model = model.to(device)
    
    requires_grad(ema, False)  # EMA 模型不需要梯度(仅用于推理)
    model = DDP(model, device_ids=[device])  # 将模型包装为分布式模型
    
    # 加载 VAE 模型(用于潜在空间和图像之间的转换)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print(f"EqM 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 准备模型进行采样:
    model.train()  # 保持训练模式以启用 embedding dropout(用于无分类器引导)
    ema.eval()  # EMA 模型应始终处于评估模式

    # 监控变量(虽然在采样中不太重要,但保留以保持代码一致性):
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # 用于条件生成的标签:
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0  # 是否使用无分类器引导(Classifier-Free Guidance)
    
    # 创建采样噪声:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # 设置无分类器引导:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)  # 复制噪声用于条件和无条件生成
        y_null = torch.tensor([1000] * n, device=device)  # 无条件标签(类别1000)
        ys = torch.cat([ys, y_null], 0)  # 拼接条件和无条件标签
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg  # 使用带 CFG 的前向传播
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward    

    # 创建输出文件夹(仅主进程)
    if rank == 0:
        os.makedirs(args.folder, exist_ok=True)
    
    # 为了使样本数能被批次大小整除,我们会多采样一些然后丢弃多余的:
    total_samples = int(math.ceil(args.num_fid_samples / args.global_batch_size) * args.global_batch_size)
    if rank == 0:
        print(f"将要采样的图像总数: {total_samples}")
    
    # 验证样本数能被进程数整除
    assert total_samples % dist.get_world_size() == 0, "total_samples 必须能被 world_size 整除"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu 必须能被每个GPU的批次大小整除"
    
    # 计算需要的迭代次数
    iterations = int(total_samples // args.global_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar  # 仅主进程显示进度条
    total = 0  # 已生成的样本总数
    n = int(args.global_batch_size // dist.get_world_size())  # 每个GPU每次生成的样本数
    
    # 开始采样循环
    for i in pbar:
        with torch.no_grad():  # 推理阶段不需要梯度
            # 1. 生成初始随机噪声(在潜在空间中)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            
            # 2. 生成随机类别标签
            y = torch.randint(0, args.num_classes, (n,), device=device)
            
            # 3. 初始化时间步 t=1 (从纯噪声开始)
            t = torch.ones((n,)).to(z).to(device)
            
            # 4. 如果使用无分类器引导,准备双份输入
            if use_cfg:
                z = torch.cat([z, z], 0)  # 复制噪声
                y_null = torch.tensor([1000] * n, device=device)  # 无条件标签
                y = torch.cat([y, y_null], 0)  # 拼接条件和无条件
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                t = torch.cat([t, t], 0)
            else:
                model_kwargs = dict(y=y)
            
            # 5. 初始化采样变量
            xt = z  # 当前状态(从噪声开始)
            m = torch.zeros_like(xt).to(xt).to(device)  # 动量项(用于 NAG-GD)
            
            # 6. 迭代去噪过程(从 t=1 逐步到 t≈0)
            for i in range(args.num_sampling_steps-1):
                if args.sampler == 'gd':
                    # 梯度下降采样器
                    # 计算当前状态下的梯度/方向
                    out = model_fn(xt, t, y, args.cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                        
                elif args.sampler == 'ngd':
                    # Nesterov 加速梯度下降采样器
                    # 先"往前看一步"再计算梯度
                    x_ = xt + args.stepsize * m * args.mu
                    out = model_fn(x_, t, y, args.cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                    m = out  # 更新动量
                
                # 更新当前状态和时间步
                xt = xt + out * args.stepsize  # 沿梯度方向移动
                t += args.stepsize  # 时间步递增
            
            # 7. 如果使用 CFG,只保留条件生成的结果
            if use_cfg:
                xt, _ = xt.chunk(2, dim=0)
            
            # 8. 使用 VAE 解码器将潜在空间转换回图像空间
            samples = vae.decode(xt / 0.18215).sample
            
            # 9. 后处理:归一化到 [0, 255] 并转换为 uint8
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # 10. 保存每张生成的图像
            for i, sample in enumerate(samples):
                # 计算全局索引(考虑多 GPU 分布式生成)
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{args.folder}/{index:06d}.png")
        
        # 更新已生成样本数
        total += args.global_batch_size
        # 同步所有进程
        dist.barrier()
    
    # 采样完成后,主进程创建 .npz 文件
    if rank == 0:
        print("正在创建 .npz 文件")
        create_npz_from_sample_folder(args.folder, 50000)
        print("完成!")
    
    cleanup()


if __name__ == "__main__":
    # 默认参数将使用我们论文中的超参数对 EqM-XL/2 进行采样
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2",
                        help="模型架构选择")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="生成图像的尺寸")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="类别数量(ImageNet 有 1000 类)")
    parser.add_argument("--global-batch-size", type=int, default=256,
                        help="全局批次大小(所有 GPU 的总和)")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="全局随机种子")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="VAE 类型选择")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="无分类器引导的缩放因子(越大越符合条件)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="训练好的 EqM 检查点路径")
    parser.add_argument("--stepsize", type=float, default=0.0017,
                        help="采样步长 eta(控制每步移动的距离)")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="采样迭代步数(越多质量越好但越慢)")
    parser.add_argument("--folder", type=str, default='samples',
                        help="保存生成图像的文件夹")
    parser.add_argument("--sampler", type=str, default='gd', choices=['gd', 'ngd'],
                        help="采样器类型: gd=梯度下降, ngd=Nesterov加速梯度下降")
    parser.add_argument("--mu", type=float, default=0.3,
                        help="NAG-GD 超参数 mu(动量系数)")
    parser.add_argument("--num-fid-samples", type=int, default=50000,
                        help="用于 FID 评估的样本数量")
    parser.add_argument("--uncond", type=bool, default=True,
                        help="禁用/启用噪声条件")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="能量公式类型")
    parse_transport_args(parser)  # 解析传输相关参数
    args = parser.parse_args()
    main(args)