import csv, math, time, warnings
from itertools import product
from pathlib import Path
from typing import Dict, Callable, List, Tuple
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import amp
from tqdm import trange, tqdm
from ptflops import get_model_complexity_info

PT_FLOPS_OK = True


DEVICE = 'cuda'
QUANT_BITS = [8, 4]
FINETUNE_EPOCHS = 1
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
CSV_PATH = Path('results/quant_sensitivity_full_metrics.csv')
CKPT_DIR = Path('checkpoints')

BASELINE_INFO = {
    "mobilenet_v3_small": {"acc": 92.32, "ckpt": "checkpoints/mobilenet_v3_small_baseline.pth"},
    "resnet50": {"acc": 95.43, "ckpt": "checkpoints/resnet50_baseline.pth"},
    "resnet152": {"acc": 95.81, "ckpt": "checkpoints/resnet152_baseline.pth"},
    "mobilenet_v3_large": {"acc": 94.67, "ckpt": "checkpoints/mobilenet_v3_large_baseline.pth"},
    "convnext_small": {"acc": 95.89, "ckpt": "checkpoints/convnext_small_baseline.pth"},
    "convnext_tiny": {"acc": 96.86, "ckpt": "checkpoints/convnext_tiny_baseline.pth"}
}

MODELS = list(BASELINE_INFO.keys())

CFG = {
    'mobilenet_v3_small': dict(bt=1024, btst=4096),
    'resnet50': dict(bt=256, btst=1024),
    'resnet152': dict(bt=128, btst=512),
    'mobilenet_v3_large': dict(bt=512, btst=2048),
    'convnext_tiny':      dict(bt=192,  btst=768),
    'convnext_small': dict(bt=96, btst=384),
}

def build_loaders(bt_train: int, bt_test: int) -> Tuple[DataLoader, DataLoader]:
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

    tr_ld = DataLoader(tr_ds, bt_train, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)
    te_ld = DataLoader(te_ds, bt_test, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=True)
    return tr_ld, te_ld


def create_model(name: str, num_cls: int = 10) -> nn.Module:
    m = getattr(models, name)(weights=None).to(DEVICE)
    if name.startswith('resnet'):
        m.fc = nn.Linear(m.fc.in_features, num_cls).to(DEVICE)
    elif name.startswith('mobilenet_v3'):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_cls).to(DEVICE)
    elif name.startswith('convnext'):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_cls).to(DEVICE)
    else:
        raise ValueError(name)
    return m

def quantizable_layers(model: nn.Module) -> List[str]:
    layers = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1) and m.out_channels >= 16:
            layers.append(n)
        elif isinstance(m, nn.Linear):
            layers.append(n)
    return layers

@torch.no_grad()
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

def finetune(model: nn.Module, loader: DataLoader, epochs: int):
    opt = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler() if DEVICE == 'cuda' else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for _ in trange(epochs, desc='ft', leave=False):
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


def _linear_quant_per_tensor(weight: torch.Tensor, bits: int, symmetric: bool):
    qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    with torch.no_grad():
        w = weight.detach()
        if symmetric:
            max_abs = w.abs().max()
            scale = max_abs / qmax if max_abs != 0 else 1.0
            q = torch.round(w / scale).clamp(-qmax - 1, qmax)
            deq = q * scale
        else:
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / qmax if w_max != w_min else 1.0
            zp = torch.round(-w_min / scale)
            q = torch.round(w / scale + zp).clamp(0, qmax)
            deq = (q - zp) * scale
        weight.data.copy_(deq)

def _linear_quant_per_channel(weight: torch.Tensor, bits: int):
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        w = weight.detach()
        out_dim = w.shape[0]
        flat = w.view(out_dim, -1)
        max_abs = flat.abs().max(dim=1).values
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
        scale = scale.view(-1, *([1] * (w.dim() - 1)))
        q = torch.round(w / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
        weight.data.copy_(deq)


def per_tensor_symmetric(m: nn.Module, bits: int):  _linear_quant_per_tensor(m.weight, bits, True)
def per_tensor_asymmetric(m: nn.Module, bits: int): _linear_quant_per_tensor(m.weight, bits, False)
def per_channel_symmetric(m: nn.Module, bits: int): _linear_quant_per_channel(m.weight, bits)

QUANT_METHODS: Dict[str, Callable[[nn.Module, int], None]] = {
    'per_tensor_symmetric': per_tensor_symmetric,
    'per_tensor_asymmetric': per_tensor_asymmetric,
    'per_channel_symmetric': per_channel_symmetric,
}


@torch.no_grad()
def inspect_model(model: nn.Module, bits: int) -> Tuple[float, float, float]:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = n_params * bits / 8 / 1024 / 1024

    if PT_FLOPS_OK:
        try:
            flops, _ = get_model_complexity_info(model.cpu(), (3, 224, 224),
                                                 as_strings=False, print_per_layer_stat=False, verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            warnings.warn(f'FLOPs calculation failed: {e}')
            flops_g = float('nan')
    else:
        flops_g = float('nan')
    model.to(DEVICE)

    reps, warm = 50, 10
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    model.eval()
    torch.cuda.empty_cache()
    if DEVICE == 'cuda':
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(warm): _ = model(dummy)
        torch.cuda.synchronize()
        elapsed = 0.0
        for _ in range(reps):
            s.record()
            _ = model(dummy)
            e.record()
            torch.cuda.synchronize()
            elapsed += s.elapsed_time(e)
        latency = elapsed / reps
    else:
        for _ in range(warm): _ = model(dummy)
        tic = time.perf_counter()
        for _ in range(reps): _ = model(dummy)
        toc = time.perf_counter()
        latency = (toc - tic) * 1000 / reps
    return size_mb, flops_g, latency

def main():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    hdr = ['model', 'layer', 'method', 'bits',
           'baseline', 'acc_no_ft', 'acc_ft', 'inst_delta', 'ft_delta',
           'model_size_MB', 'flops_G', 'latency_ms']

    with CSV_PATH.open('w', newline='') as f_csv:
        wr = csv.DictWriter(f_csv, fieldnames=hdr)
        wr.writeheader()
        f_csv.flush()

        for mdl in MODELS:
            info = BASELINE_INFO.get(mdl)
            ckpt_path = Path(info['ckpt'])
            baseline_acc = info['acc']
            cfg = CFG[mdl]
            tr_loader, te_loader = build_loaders(cfg['bt'], cfg['btst'])

            model = create_model(mdl)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.to(DEVICE).train(False)
            baseline_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            layers = quantizable_layers(model)
            tasks = list(product(QUANT_METHODS.items(), layers, QUANT_BITS))

            print(f'\n=== {mdl} | baseline_acc = {baseline_acc:.2f}% | tasks={len(tasks)} ===')
            for (meth_name, q_fn), layer, bits in tqdm(tasks, desc=f'{mdl}-tasks'):
                model.load_state_dict(baseline_wts, strict=True)
                tgt = dict(model.named_modules())[layer]
                q_fn(tgt, bits)

                size_mb, flops_g, lat_ms = inspect_model(model, bits)
                acc_no_ft = evaluate(model, te_loader)
                if bits <= 4:
                    finetune(model, tr_loader, FINETUNE_EPOCHS)
                    acc_ft = evaluate(model, te_loader)
                else:
                    acc_ft = acc_no_ft

                wr.writerow(dict(
                    model=mdl, layer=layer, method=meth_name, bits=bits,
                    baseline=baseline_acc, acc_no_ft=acc_no_ft, acc_ft=acc_ft,
                    inst_delta=acc_no_ft - baseline_acc,
                    ft_delta=acc_ft - baseline_acc,
                    model_size_MB=round(size_mb, 3),
                    flops_G=round(flops_g, 3) if not math.isnan(flops_g) else 'nan',
                    latency_ms=round(lat_ms, 3)
                ))
            f_csv.flush()

if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()

