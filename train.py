# 本源代码的许可证位于根目录的 LICENSE 文件中

"""
使用 PyTorch DDP 的 EqM 最小训练脚本
EqM: Equivariant Model - 等变模型,用于图像生成任务
"""
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
import wandb_utils  # Weights & Biases 日志工具
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F

#################################################################################
#                             训练辅助函数                                        #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    将 EMA (指数移动平均) 模型朝当前模型方向更新
    
    EMA 是一种模型权重平滑技术,可以提高模型的稳定性和泛化能力
    
    参数:
        ema_model: 指数移动平均模型
        model: 当前训练的模型
        decay: 衰减率,默认 0.9999,值越大历史权重保留越多
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: 考虑仅对 require_grad 的参数应用,以避免 pos_embed 的小数值变化
        # EMA 更新公式: ema_param = decay * ema_param + (1 - decay) * current_param
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


def create_logger(logging_dir):
    """
    创建一个同时写入日志文件和标准输出的日志记录器
    
    参数:
        logging_dir: 日志文件保存目录
        
    返回:
        logger: 日志记录器对象
    """
    if dist.get_rank() == 0:  # 真实的日志记录器(仅主进程)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',  # 带颜色的时间戳
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # 虚拟日志记录器(其他进程不输出)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    中心裁剪实现,来自 ADM (Ablated Diffusion Model)
    
    处理流程:
    1. 首先将图像缩小到合适大小(使用 BOX 滤波器)
    2. 然后缩放到目标尺寸(使用双三次插值)
    3. 最后进行中心裁剪
    
    参数:
        pil_image: PIL 图像对象
        image_size: 目标图像尺寸
        
    返回:
        裁剪后的 PIL 图像
    
    参考: https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # 如果图像最小边大于目标尺寸的2倍,先缩小一半(快速降采样)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 计算缩放比例,使最小边等于 image_size
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 转换为 numpy 数组并进行中心裁剪
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  训练循环                                      #
#################################################################################

def main(args):
    """
    训练新的 EqM 模型的主函数
    
    参数:
        args: 命令行参数对象,包含所有训练配置
    """
    # 验证 CUDA 可用性
    assert torch.cuda.is_available(), "训练当前至少需要一个 GPU。"
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
    
    # 设置随机种子(每个进程使用不同的种子以确保数据多样性)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"启动 rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # 计算每个进程的批次大小
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # 设置实验文件夹(仅主进程):
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # 创建结果文件夹
        experiment_index = len(glob(f"{args.results_dir}/*"))  # 实验编号
        model_string_name = args.model.replace("/", "-")  # 将模型名中的 / 替换为 -
        # 生成实验名称
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # 实验文件夹
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # 检查点文件夹
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"实验目录创建于 {experiment_dir}")

        # 初始化 Weights & Biases 日志
        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

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
    ema = deepcopy(model).to(device)  # 创建模型的 EMA 副本,用于训练后使用
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # 如果提供了检查点,加载模型权重
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
            opt.load_state_dict(state_dict["opt"])
        else:
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)

        ema = ema.to(device)
        model = model.to(device)
    
    requires_grad(ema, False)  # EMA 模型不需要梯度
    model = DDP(model, device_ids=[device])  # 将模型包装为分布式模型
    
    # 创建传输对象(用于扩散过程)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        alpha=args.alpha,          # 传递
        lambda_val=args.lambda_val # 传递
    )
    transport_sampler = Sampler(transport)
    
    # 加载 VAE 模型(用于图像和潜在空间之间的转换)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 设置数据加载:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转(数据增强)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  # 归一化到 [-1, 1]
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    
    # 分布式采样器,确保每个进程处理不同的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,  # 使用 sampler 时不能 shuffle
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,  # 将数据固定在内存中,加速 GPU 传输
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    logger.info(f"数据集包含 {len(dataset):,} 张图像 ({args.data_path})")

    # 准备模型进行训练:
    update_ema(ema, model.module, decay=0)  # 确保 EMA 使用同步的权重初始化
    model.train()  # 重要!这会启用 embedding dropout 用于无分类器引导
    ema.eval()  # EMA 模型应始终处于评估模式

    # 用于监控/日志记录的变量:
    train_steps = 0  # 训练步数
    log_steps = 0  # 日志步数
    running_loss = 0  # 累计损失
    start_time = time()

    # 用于条件生成的标签(可自由更改):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0  # 是否使用无分类器引导(Classifier-Free Guidance)
    
    # 创建采样噪声:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # 设置无分类器引导:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)  # 复制噪声
        y_null = torch.tensor([1000] * n, device=device)  # 无条件标签(类别1000表示无条件)
        ys = torch.cat([ys, y_null], 0)  # 拼接条件和无条件标签
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg  # 使用带 CFG 的前向传播
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward
    
    logger.info(f"训练 {args.epochs} 个 epoch...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 设置 epoch 以确保不同 epoch 的数据洗牌不同
        logger.info(f"开始 epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                # 将输入图像映射到潜在空间并归一化潜在变量:
                # 0.18215 是 Stable Diffusion VAE 的缩放因子
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            # 准备模型输入参数
            model_kwargs = dict(y=y, return_act=args.disp, train=True)
            
            # 计算训练损失
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # 反向传播和优化
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)  # 更新 EMA 模型

            # 记录损失值:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # 定期记录训练信息
            if train_steps % args.log_every == 0:
                # 测量训练速度:
                torch.cuda.synchronize()  # 同步 GPU 操作
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # 在所有进程间归约损失历史:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)  # 求和所有进程的损失
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                logger.info(f"(step={train_steps:07d}) 训练损失: {avg_loss:.4f}, 训练步数/秒: {steps_per_sec:.2f}")
                
                # 记录到 Weights & Biases
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                
                # 重置监控变量:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 保存 EqM 检查点:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:  # 仅主进程保存
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"检查点已保存到 {checkpoint_path}")
                dist.barrier()  # 同步所有进程
                
    model.eval()  # 重要!这会禁用随机化的 embedding dropout
    # 使用 ema(或 model)在评估模式下进行任何采样/FID计算等...

    logger.info("完成!")
    cleanup()


if __name__ == "__main__":
    # 默认参数将使用我们论文中的超参数训练 EqM-XL/2(除了训练迭代次数)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="结果保存目录")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2",
                        help="模型架构选择")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="图像尺寸")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="类别数量")
    parser.add_argument("--epochs", type=int, default=80,
                        help="训练轮数")
    parser.add_argument("--global-batch-size", type=int, default=256,
                        help="全局批次大小")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="全局随机种子")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="VAE 类型选择(不影响训练)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载的工作进程数")
    parser.add_argument("--log-every", type=int, default=100,
                        help="每多少步记录一次日志")
    parser.add_argument("--ckpt-every", type=int, default=50000,
                        help="每多少步保存一次检查点")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="无分类器引导的缩放因子")
    parser.add_argument("--wandb", action="store_true",
                        help="是否启用 Weights & Biases 日志")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="可选的自定义 EqM 检查点路径")
    parser.add_argument("--disp", action="store_true",
                        help="切换以启用分散损失(Dispersive Loss)")
    parser.add_argument("--uncond", type=bool, default=True,
                        help="禁用/启用噪声条件")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="能量公式类型")
    parser.add_argument("--fts-alpha", type=float, default=0.5, help="FTS alpha parameter")
    parse_transport_args(parser)  # 解析传输相关参数
    args = parser.parse_args()
    main(args)