import torch
import argparse
from models import EqM_models
from download import find_model  # 和你推理脚本一致

def pretty_print_dict(d, indent=0):
    """辅助打印字典层级结构"""
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"- {k} (dict, {len(v)} keys)")
        elif isinstance(v, torch.Tensor):
            print("  " * indent + f"- {k}: Tensor{tuple(v.shape)}")
        else:
            print("  " * indent + f"- {k}: {type(v)}")

def inspect_state_dict(name, sd):
    print(f"\n========== 🧩 Inspecting {name} ==========")
    total_params = 0
    for key, value in sd.items():
        total_params += value.numel()

    print(f"{name} 总参数量: {total_params:,}")

    # 打印前 10 层的权重形状，避免过多
    print(f"{name} 中前 10 个参数的形状:")
    for i, (key, value) in enumerate(sd.items()):
        if i >= 10:
            print("... (更多参数省略)")
            break
        print(f"  {key:<40} {tuple(value.shape)}")

        
def test_ckpt(ckpt_path):
    print(f"\n==== 读取 checkpoint: {ckpt_path} ====\n")

    # -------------------------------
    # ① 加载 pt 文件（自动支持 URL / 本地）
    # -------------------------------
    try:
        ckpt = find_model(ckpt_path)
    except Exception as e:
        print("❌ find_model 加载失败，请检查路径或 URL")
        print(e)
        return

    print("\n==== >> checkpoint.keys(): ")
    print(list(ckpt.keys()))
    print()

    # -------------------------------
    # ② 判断格式：完整 checkpoint？还是 state_dict？
    # -------------------------------
    if "model" in ckpt and "ema" in ckpt:
        print("📦 检测到 **完整 checkpoint 格式**（包含 model + ema）")
        print("model keys:")
        print(list(ckpt["model"].keys())[:10], "...")

        print("\nema keys:")
        print(list(ckpt["ema"].keys())[:10], "...\n")

        print("参数数量：")
        print(" - model 共有", len(ckpt["model"]), "个参数")
        print(" - ema    共有", len(ckpt["ema"]), "个参数")

    else:
        print("📦 检测到 **纯 state_dict 格式**（通常只有 EMA 权重）")
        print("state_dict keys:")
        print(list(ckpt.keys())[:10], "...")

        print("\n参数数量：", len(ckpt), "个参数\n")

    if "opt" in ckpt:
        print("\n📎 发现优化器状态 opt:")
        print(f"opt 的 key 数量 = {len(ckpt['opt'])}")
    elif "optimizer" in ckpt:
        print("\n📎 发现优化器状态 optimizer:")
        print(f"optimizer 的 key 数量 = {len(ckpt['optimizer'])}")
    else:
        print("\n⚪ 未发现优化器状态（opt / optimizer）")

    print("\n========== 🔍 Checkpoint Keys ==========")
    print(list(ckpt.keys()))
    # -------------------------------
    # 2. Inspect optimizer state
    # -------------------------------
    if "opt" in ckpt:
        opt = ckpt["opt"]
        print("\n========== ⚙️ Inspecting Optimizer (opt) ==========")

        # optimizer"type" 信息（如果有的话）
        if "param_groups" in opt:
            pg = opt["param_groups"][0]
            print("\n👉 Optimizer 参数组:")
            print(f" - lr                = {pg.get('lr', 'N/A')}")
            print(f" - betas             = {pg.get('betas', 'N/A')}")
            print(f" - weight_decay      = {pg.get('weight_decay', 'N/A')}")
            print(f" - eps               = {pg.get('eps', 'N/A')}")
            print(f" - maximize          = {pg.get('maximize', 'N/A')}")

        # 优化器内部状态（例如 Adam 的 m, v）
        if "state" in opt:
            states = list(opt["state"].items())
            print(f"\n👉 Optimizer 内 state 数量: {len(states)}")

            # 打印前 2 个 state 以免过多
            for i, (pid, st) in enumerate(states[:2]):
                print(f"\n参数 ID = {pid}")
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k:<20} tensor shape = {tuple(v.shape)}")
                    else:
                        print(f"  {k:<20} value = {v}")

            if len(states) > 2:
                print("... (更多 state 省略)")

    # -------------------------------
    # 3. Inspect training step
    # -------------------------------
    if "step" in ckpt:
        print(f"\n========== 📈 Training Step ==========")
        print(f"当前训练 step = {ckpt['step']}")
    else:
        print("\n📭 未找到训练 step 信息（可能是纯权重文件）")


    opt_state = ckpt.get("opt", None)

    if opt_state is None:
        print("❌ checkpoint 中没有 opt（优化器状态），无法读取 step")
        exit(0)

    state = opt_state["state"]

    steps = []

    print("\n===== 🔍 打印每个参数的 Adam step =====\n")

    for param_id, param_state in state.items():
        step = param_state.get("step", None)

        if step is None:
            print(f"参数 ID {param_id}: ❌ 没有 step 字段")
            continue

        # 有些 step 是普通 int，有些是 0-dim tensor
        if torch.is_tensor(step):
            step = step.item()

        print(f"参数 ID {param_id}: step = {step}")
        steps.append(step)

    # 汇总信息
    if steps:
        print("\n===== 📊 Step 统计信息 =====")
        print(f"参数数量: {len(steps)}")
        print(f"最小 step: {min(steps)}")
        print(f"最大 step: {max(steps)}")
        print(f"平均 step: {sum(steps) / len(steps):.2f}")

        print("\n⚠️（一般来说 max_step ≈ 实际训练步数）")
        print("\n🎉 checkpoint 文件格式检查完成\n")
    
    model_sd = ckpt["model"]
    ema_sd = ckpt["ema"]

    param_name = "blocks.0.attn.qkv.weight"

    print("\n===== Model 参数值 =====\n")
    print(model_sd[param_name])

    print("\n===== EMA 参数值 =====\n")
    print(ema_sd[param_name])
    param_name = "blocks.0.attn.qkv.weight"

    w_m = model_sd[param_name]
    w_e = ema_sd[param_name]

    print("\n===== 参数对比 =====")
    print("Model weight shape:", w_m.shape)
    print("EMA weight shape:  ", w_e.shape)

    print("\n--- Model 前 5 个值 ---")
    print(w_m.flatten()[:5])

    print("\n--- EMA 前 5 个值 ---")
    print(w_e.flatten()[:5])

    print("\n--- 差值 (EMA - Model) 前 5 个 ---")
    print((w_e - w_m).flatten()[:5])

    print("\n差值 L2 范数 =", torch.norm(w_e - w_m).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="pt 文件路径或 URL")
    args = parser.parse_args()

    test_ckpt(args.ckpt)
