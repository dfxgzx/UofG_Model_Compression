import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Dict, Callable, List, Tuple

import torch, torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import amp
from tqdm import trange

DEVICE = 'cuda'
PRUNE_AMOUNTS = [0.9, 0.8]
BASELINE_EPOCHS = 8
FINETUNE_EPOCHS = 1
BASE_LR = 3e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
CSV_PATH = Path('results/pruned_by_layer_cumulative_to_middle.csv')

torch.manual_seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

CFG = {
    'mobilenet_v3_small': dict(bt=1024, btst=4096),
    'resnet50':           dict(bt=256,  btst=1024),
    'resnet152':          dict(bt=128,  btst=512),
    'mobilenet_v3_large': dict(bt=512,  btst=2048),
    'convnext_tiny':      dict(bt=192,  btst=768),
    'convnext_small':     dict(bt=96,   btst=384),
}
MODELS = list(CFG.keys())

def build_loaders(batch_train: int, batch_test: int) -> Tuple[DataLoader, DataLoader]:
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    te_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    tr_ds = datasets.CIFAR10('data', train=True, download=True, transform=tr_tf)
    te_ds = datasets.CIFAR10('data', train=False, download=True, transform=te_tf)
    tr_ld = DataLoader(tr_ds, batch_train, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)
    te_ld = DataLoader(te_ds, batch_test, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=True)
    return tr_ld, te_ld

def create_model(name: str, num_cls: int = 10) -> nn.Module:
    m = getattr(models, name)(weights='DEFAULT').to(DEVICE)
    if name.startswith('resnet'):
        m.fc = nn.Linear(m.fc.in_features, num_cls).to(DEVICE)
    elif name.startswith('mobilenet_v3'):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_cls).to(DEVICE)
    elif name.startswith('convnext'):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_cls).to(DEVICE)
    else:
        raise ValueError(name)
    return m

def prunable_layers(model: nn.Module) -> List[str]:
    layers = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1) and m.out_channels >= 16:
            layers.append(n)
        elif isinstance(m, nn.Linear):
            layers.append(n)
    return layers

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    tot = correct = 0
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with amp.autocast(device_type='cuda', dtype=dtype):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
    return 100. * correct / tot

def finetune(model: nn.Module, loader: DataLoader,
             epochs: int, lr: float, desc: str):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler() if DEVICE == 'cuda' else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    for _ in trange(epochs, desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', dtype=dtype):
                loss = loss_fn(model(x), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

def random_prune(m, amt): prune.random_unstructured(m, 'weight', amt)
def l1_prune(m, amt):     prune.l1_unstructured(m, 'weight', amt)
def l2_structured_prune(m, amt): prune.ln_structured(m, 'weight', amt, n=2, dim=0)

PRUNE_METHODS: Dict[str, Callable[[nn.Module, float], None]] = {
    'l1_unstructured': l1_prune,
    'random_unstructured': random_prune,
    'l2_structured': l2_structured_prune,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=None,
                        help='Override default (continue from middle to full).')
    args = parser.parse_args()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ['model', 'layer', 'method', 'ratio',
              'baseline', 'acc_no_ft', 'acc_ft',
              'inst_delta', 'ft_delta', 'params_pruned (%)']

    existing_baselines = {}
    if CSV_PATH.exists():
        with CSV_PATH.open('r') as f_prev:
            reader = csv.DictReader(f_prev)
            for row in reader:
                mdl = row['model']
                if mdl not in existing_baselines:
                    existing_baselines[mdl] = float(row['baseline'])

    with CSV_PATH.open('a', newline='') as f_csv:
        wr = csv.DictWriter(f_csv, fieldnames=header)
        if CSV_PATH.stat().st_size == 0:
            wr.writeheader()
            f_csv.flush()

        for mdl_name in MODELS:
            cfg = CFG[mdl_name]
            tr_loader, te_loader = build_loaders(cfg['bt'], cfg['btst'])
            print(f'\n=== {mdl_name} (continue to full) ===')

            model = create_model(mdl_name)
            baseline_acc = existing_baselines[mdl_name]
            print(f'[SKIP base] Using cached baseline = {baseline_acc:.2f}%')
            finetune(model, tr_loader, 0, BASE_LR, desc="dry")
            n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            baseline_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            all_layers = prunable_layers(model)
            total_layers = len(all_layers)

            for depth_level in range(1, total_layers + 1):
                layers_k = all_layers[-depth_level:]
                layer_tag = '|'.join(layers_k)

                for (meth_name, prune_fn), ratio in product(PRUNE_METHODS.items(), PRUNE_AMOUNTS):
                    model.load_state_dict(baseline_wts, strict=True)

                    pruned_params = 0
                    for ln in layers_k:
                        mod = dict(model.named_modules())[ln]
                        prune_fn(mod, amt=ratio)
                        pruned_params += int(ratio * mod.weight.numel())

                    acc_no_ft = evaluate(model, te_loader)

                    if ratio >= 0.5:
                        finetune(model, tr_loader, FINETUNE_EPOCHS, FINETUNE_LR, desc="ft")
                        acc_ft = evaluate(model, te_loader)
                    else:
                        acc_ft = acc_no_ft

                    wr.writerow(dict(
                        model=mdl_name,
                        layer=layer_tag,
                        method=meth_name,
                        ratio=ratio,
                        baseline=baseline_acc,
                        acc_no_ft=acc_no_ft,
                        acc_ft=acc_ft,
                        inst_delta=acc_no_ft - baseline_acc,
                        ft_delta=acc_ft - baseline_acc,
                        **{'params_pruned (%)': round(100. * pruned_params / n_params_total, 4)}
                    ))
                    f_csv.flush()

                    for ln in layers_k:
                        prune.remove(dict(model.named_modules())[ln], 'weight')

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

