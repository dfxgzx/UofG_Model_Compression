import csv
import json
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch import amp
from tqdm import trange, tqdm

DEVICE = 'cuda'

CONFIG_CSV_PATH = Path('vgg16_compression_strategy_.csv')

BASELINE_CKPT_PATH = Path('checkpoints/vgg16_baseline.pth')
BASELINE_ACC_JSON = Path('baseline_acc.json')
FINAL_MODEL_SAVE_PATH = Path('vgg16_mixed_strategy_final.pth')
FINETUNE_EPOCHS = 2
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 512
NUM_WORKERS = 8

torch.manual_seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def build_loaders() -> Tuple[DataLoader, DataLoader]:
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    te_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_ds = datasets.CIFAR10('data', train=True, download=True, transform=tr_tf)
    te_ds = datasets.CIFAR10('data', train=False, download=True, transform=te_tf)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE_TRAIN, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None)
    te_ld = DataLoader(te_ds, BATCH_SIZE_TEST, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=(NUM_WORKERS > 0))
    return tr_ld, te_ld


def create_vgg16(num_classes: int = 10) -> nn.Module:
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model.to(DEVICE)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, desc: str = "Evaluating") -> float:
    model.eval()
    total = correct = 0
    use_amp = (DEVICE == 'cuda')
    dtype = torch.bfloat16 if use_amp else torch.float32
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        if use_amp:
            with amp.autocast(device_type='cuda', dtype=dtype):
                logits = model(x)
        else:
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100. * correct / total


def finetune(model: nn.Module, loader: DataLoader, epochs: int, lr: float, desc: str):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = (DEVICE == 'cuda')
    dtype = torch.bfloat16 if use_amp else torch.float32

    for epoch in trange(epochs, desc=desc):
        model.train()
        for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with amp.autocast(device_type='cuda', dtype=dtype):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()


def is_prunable_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight')


def l1_prune(module: nn.Module, amount: float):
    if amount > 0:
        prune.l1_unstructured(module, 'weight', amount)


def per_tensor_symmetric_quant(module: nn.Module, bits: int):
    weight = module.weight
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        w = weight.detach()
        max_abs = w.abs().max()
        scale = max_abs / qmax if max_abs != 0 else 1.0
        q = torch.round(w / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
        weight.data.copy_(deq)


def read_config_from_csv(csv_path: Path) -> Dict[str, Dict]:
    config = {}
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            layer_name = row['layer_name'].strip()
            if '|' in layer_name:
                print(f"Skip merged layer entry '{layer_name}'. Please provide single-layer rows.")
                continue
            config[layer_name] = {
                'pruning_ratio': float(row['assigned_pruning_ratio']),
                'quantization_bits': int(row['assigned_quantization_bits'])
            }
    print(f"Successfully read configuration for {len(config)} layers from {csv_path}")
    return config


def main():
    strategy_config = read_config_from_csv(CONFIG_CSV_PATH)
    tr_loader, te_loader = build_loaders()
    model = create_vgg16()
    if not BASELINE_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at: {BASELINE_CKPT_PATH}\n"
            "Please train a VGG16 baseline model first and save its weights."
        )
    model.load_state_dict(torch.load(BASELINE_CKPT_PATH, map_location=DEVICE))

    with open(BASELINE_ACC_JSON, 'r') as f:
        baseline_acc = json.load(f).get("vgg16", 0.0)
    print(f"Loaded baseline accuracy: {baseline_acc:.2f}%")

    print("\nApplying mixed-strategy (Pruning + Quantization) on VGG16...")
    modules = dict(model.named_modules())

    pruned_layers = []
    for layer_name, cfg in tqdm(strategy_config.items(), desc="Applying Pruning"):
        m = modules.get(layer_name)
        if m and is_prunable_module(m) and cfg['pruning_ratio'] > 0:
            l1_prune(m, cfg['pruning_ratio'])
            pruned_layers.append(layer_name)

    print(f"Applied pruning to {len(pruned_layers)} layers.")

    for layer_name in tqdm(pruned_layers, desc="Making Pruning Permanent"):
        prune.remove(modules[layer_name], 'weight')
    print("Pruning made permanent.")

    quantized_layers_count = 0
    for layer_name, cfg in tqdm(strategy_config.items(), desc="Applying Quantization"):
        m = modules.get(layer_name)
        if m and is_prunable_module(m):
            per_tensor_symmetric_quant(m, cfg['quantization_bits'])
            quantized_layers_count += 1

    print(f"Applied quantization to {quantized_layers_count} layers.")

    acc_no_ft = evaluate(model, te_loader, "Eval (No Finetune)")

    finetune(model, tr_loader, FINETUNE_EPOCHS, FINETUNE_LR, "Finetuning")

    acc_ft = evaluate(model, te_loader, "Eval (After Finetune)")

    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f"\nFinal compressed model state dictionary saved to: {FINAL_MODEL_SAVE_PATH}")

    print("\n" + "=" * 50)
    print("Mixed-Strategy Compression Results (VGG16)")
    print("=" * 50)
    print(f"Baseline Accuracy:{baseline_acc:.2f}%")
    print(f"Accuracy (No Finetune):{acc_no_ft:.2f}%")
    print(f"Accuracy (After {FINETUNE_EPOCHS} Epochs FT): {acc_ft:.2f}%")
    print("-" * 50)
    print(f"Instant Accuracy Drop:{acc_no_ft - baseline_acc:+.2f}%")
    print(f"Final Accuracy Change vs Base:{acc_ft - baseline_acc:+.2f}%")
    print(f"Accuracy Recovered by FT:{acc_ft - acc_no_ft:+.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

