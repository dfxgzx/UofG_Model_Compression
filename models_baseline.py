import json
from pathlib import Path
from typing import Tuple
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import amp
from tqdm import trange

DEVICE = 'cuda'
BASELINE_EPOCHS = 8
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True, parents=True)

CFG = {
    'mobilenet_v3_small': dict(bt=1024, btst=4096),
    'resnet50': dict(bt=256, btst=1024),
    'resnet152': dict(bt=128, btst=512),
    'mobilenet_v3_large': dict(bt=512, btst=2048),
    'convnext_tiny': dict(bt=192, btst=768),
    'convnext_small': dict(bt=96, btst=384),
}
MODELS = list(CFG.keys())

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

def finetune(model: nn.Module, loader: DataLoader):
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler() if DEVICE == 'cuda' else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for _ in trange(BASELINE_EPOCHS, desc='baseline-train'):
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

if __name__ == '__main__':
    torch.manual_seed(42)
    results = {}
    for mdl in MODELS:
        print(f'\n=== Baseline: {mdl} ===')
        cfg = CFG[mdl]
        tr_loader, te_loader = build_loaders(cfg['bt'], cfg['btst'])
        model = create_model(mdl)

        finetune(model, tr_loader)
        acc = evaluate(model, te_loader)

        print(f'baseline_acc = {acc:.2f}%')
        results[mdl] = acc
        torch.save(model.state_dict(), CKPT_DIR / f'{mdl}_baseline.pth')

    print('\n=== SUMMARY ===')
    for k, v in results.items():
        print(f'{k}: {v:.2f}%')

    with open('baseline_acc.json', 'w') as f:
        json.dump(results, f, indent=2)

