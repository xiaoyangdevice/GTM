
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import pandas as pd
import argparse
import warnings
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from glue2 import (
    DEVICE, VOCAB_SIZE, tokenizer, CHECKPOINT_DIR,
    PureTransformer, PureMamba, GatedTransMamba, Hyena, LSTM_Attn, AlternateTransMamba,
    RetFormer, GatedMLP, RWKV, RetNet, TransMamba,
    train_epoch, evaluate, count_parameters, get_cosine_schedule_with_warmup,
    load_pretrained_weights, compute_flops
)

warnings.filterwarnings('ignore')

# ==============================================================================
# MNLI 配置
# ==============================================================================
BATCH_SIZE = 16
GRAD_ACCUM = 2
MAX_LEN = 256  # MNLI 是句子对任务，需要更长序列
EPOCHS = 5     # MNLI 数据集较大，减少epoch数
EPOCHS_OURS = 5
LR = 2e-5
LR_OURS = 2e-5
OUTPUT_DIR = Path("./")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型名称映射（带池化方式配置）
def get_model_class(model_name):
    """获取模型类和默认配置"""
    model_configs = {
        "Ours": (GatedTransMamba, {"pooling": "mean", "use_bidirectional": True}),
       "Ours-CLS": (GatedTransMamba, {"pooling": "cls", "use_bidirectional": True}),
        # "Ours-Max": (GatedTransMamba, {"pooling": "max", "use_bidirectional": True}),
        # "Ours-Concat": (GatedTransMamba, {"pooling": "concat", "use_bidirectional": True}),
        # "Ours-Uni": (GatedTransMamba, {"pooling": "mean", "use_bidirectional": False}),  # 单向Mamba对照
        "Alternate": (AlternateTransMamba, {"use_bidirectional": True}),
        "Transformer": (PureTransformer, {}),
        "Mamba": (PureMamba, {"use_bidirectional": True}),
        # "Mamba-Uni": (PureMamba, {"use_bidirectional": False}),  # 单向Mamba对照
        "RetFormer": (RetFormer, {}),
        "GatedMLP": (GatedMLP, {}),
        "RWKV": (RWKV, {"bidirectional": True}),
        "RetNet": (RetNet, {"bidirectional": True}),
        "TransMamba": (TransMamba, {"use_bidirectional": True}),  # 序列级别分段混合（新增）
    }
    return model_configs.get(model_name, (None, {}))


def load_mnli():
    """加载 MNLI 数据集"""
    print("📂 Loading MNLI dataset...")
    
    def tokenize(ex):
        return tokenizer(
            ex["premise"], 
            ex["hypothesis"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN
        )
    
    ds = load_dataset("glue", "mnli")
    train_ds = ds["train"].map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    # MNLI 有两个验证集：validation_matched 和 validation_mismatched
    val_matched_ds = ds["validation_matched"].map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    val_mismatched_ds = ds["validation_mismatched"].map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    
    train_ds.set_format("torch", columns=["input_ids", "label"])
    val_matched_ds.set_format("torch", columns=["input_ids", "label"])
    val_mismatched_ds.set_format("torch", columns=["input_ids", "label"])
    
    print(f"  训练样本: {len(train_ds)}")
    print(f"  验证样本(matched): {len(val_matched_ds)}")
    print(f"  验证样本(mismatched): {len(val_mismatched_ds)}")
    
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_matched_ds, batch_size=BATCH_SIZE),
        DataLoader(val_mismatched_ds, batch_size=BATCH_SIZE)
    )


def run_single_model(model_name, train_loader, val_matched_loader, val_mismatched_loader, 
                     use_pretrained=True, pooling="mean"):
   
    print(f"\n{'='*60}")
    print(f"📊 Training: {model_name}")
    print(f"{'='*60}")
    
    # 获取模型类和配置
    ModelClass, model_config = get_model_class(model_name)
    
    if ModelClass is None:
        print(f"❌ 未知模型: {model_name}")
        return {
            "Model": model_name, 
            "Acc-Matched(%)": "N/A", 
            "Acc-Mismatched(%)": "N/A",
            "Mem(GB)": "N/A", 
            "FLOPs(G)": "N/A"
        }
    
    # 根据模型类型设置超参数
    if model_name.startswith("Ours"):
        epochs = EPOCHS_OURS
        lr = LR_OURS
        weight_decay = 0.01
        dropout = 0.1
    else:
        epochs = EPOCHS
        lr = LR
        weight_decay = 0.01
        dropout = 0.1
    
    # 创建模型 - MNLI是三分类任务
    model = ModelClass(num_classes=3, dropout=dropout, **model_config).to(DEVICE)
    print(f"参数量: {count_parameters(model)/1e6:.2f}M")
    print(f"模型配置: {model_config}")
    
    # 计算 FLOPs
    flops = compute_flops(model, input_seq_len=MAX_LEN, batch_size=1)
    print(f"FLOPs: {flops:.2f} GFLOPs")
    
    # 加载预训练权重
    if use_pretrained:
        print(f"\n🔄 加载预训练权重...")
        # 对于变体模型，尝试加载基础模型的权重
        pretrained_name = "Ours" if model_name.startswith("Ours") else model_name
        model, loaded = load_pretrained_weights(model, pretrained_name, CHECKPOINT_DIR)
        if loaded:
            print(f"  ✅ 预训练权重加载成功")
        else:
            print(f"  ⚠️ 未加载预训练权重，从头训练")
    
    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 损失函数
    crit = nn.CrossEntropyLoss()
    
    # GradScaler
    scaler = torch.amp.GradScaler('cuda')
    
    # 学习率调度器
    total_steps = (len(train_loader) // GRAD_ACCUM) * epochs
    scheduler = get_cosine_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)
    
    best_acc_matched, best_acc_mismatched = 0, 0
    final_mem = 0
    
    for epoch in range(epochs):
        # 训练
        loss, mem, speed = train_epoch(model, train_loader, opt, crit, scaler, scheduler, GRAD_ACCUM)
        
        # 评估 - matched
        acc_matched, f1_matched, lat, ppl = evaluate(model, val_matched_loader)
        # 评估 - mismatched
        acc_mismatched, f1_mismatched, lat, ppl = evaluate(model, val_mismatched_loader)
        
        final_mem = mem
        
        # 更新最佳结果（使用matched准确率作为主要指标）
        if acc_matched > best_acc_matched:
            best_acc_matched = acc_matched
            best_acc_mismatched = acc_mismatched
            # 保存最佳模型
            save_path = OUTPUT_DIR / f"{model_name}_mnli_best.pt"
            torch.save(model.state_dict(), save_path)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | "
              f"Acc-Matched: {acc_matched:.4f} | Acc-Mismatched: {acc_mismatched:.4f} | "
              f"Best-Matched: {best_acc_matched:.4f}")
    
    # 清理
    del model, opt, scaler
    torch.cuda.empty_cache()
    
    return {
        "Model": model_name,
        "Acc-Matched(%)": round(best_acc_matched * 100, 2),
        "Acc-Mismatched(%)": round(best_acc_mismatched * 100, 2),
        "Mem(GB)": round(final_mem, 2),
        "FLOPs(G)": round(flops, 2)
    }


def run_mnli_experiment(use_pretrained=True, pooling="mean", models=None):
   
    print("\n" + "="*60)
    print("🚀 MNLI Classification Experiment")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Use Pretrained: {use_pretrained}")
    print(f"Default Pooling: {pooling}")
    print("="*60)
    
    # 检查可用的检查点
    if use_pretrained and CHECKPOINT_DIR.exists():
        print("\n📁 可用的预训练检查点:")
        for model_dir in CHECKPOINT_DIR.iterdir():
            if model_dir.is_dir():
                best_pt = model_dir / "checkpoint_best.pt"
                status = "✅" if best_pt.exists() else "❌"
                print(f"  {status} {model_dir.name}")
    
    # 加载数据
    train_loader, val_matched_loader, val_mismatched_loader = load_mnli()
    
    # 默认模型列表
    if models is None:
        models = [
            "Ours",           # GatedTransMamba + mean池化 + 双向Mamba
            "Ours-CLS",       # GatedTransMamba + CLS池化 + 双向Mamba
            "Alternate",      # 交替混合
            "Transformer",    # 纯Transformer
            "Mamba",          # 纯Mamba（双向）
            "RetFormer",      # RetFormer
            "GatedMLP",       # GatedMLP
            "RWKV",           # RWKV（双向）
            "RetNet",         # RetNet（双向）
            "TransMamba",     # 序列级别分段混合（新增）
        ]
    
    # 训练所有模型
    results = []
    for model_name in models:
        try:
            result = run_single_model(
                model_name, 
                train_loader, val_matched_loader, val_mismatched_loader,
                use_pretrained=use_pretrained,
                pooling=pooling
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "Model": model_name,
                "Acc-Matched(%)": "Error",
                "Acc-Mismatched(%)": "Error",
                "Mem(GB)": "Error",
                "FLOPs(G)": "Error"
            })
    
    # 输出结果
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("✅ MNLI Results")
    print("="*60)
    print(df.to_string(index=False))
    
    # 保存结果
    result_file = OUTPUT_DIR / "mnli_results1.csv"
    df.to_csv(result_file, index=False)
    print(f"\n📄 结果已保存: {result_file}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNLI Classification Experiment")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR), help="检查点目录")
    parser.add_argument("--pooling", type=str, default="mean", 
                        choices=["cls", "mean", "max", "concat"],
                        help="池化方式（默认mean）")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="要训练的模型列表，例如: Ours Transformer Mamba")
    args = parser.parse_args()
    
    # 更新检查点目录
    if args.checkpoint_dir:
        CHECKPOINT_DIR = Path(args.checkpoint_dir)
    
    run_mnli_experiment(
        use_pretrained=not args.no_pretrained,
        pooling=args.pooling,
        models=args.models
    )