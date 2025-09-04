import csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda'

BASELINE_MODEL_PATH = Path('checkpoints/vgg16_baseline.pth')
COMPRESSED_MODEL_PATH = Path('checkpoints/vgg16_mixed_strategy_final_v2.pth')

REPORT_CSV_PATH = Path('results/compression_comparison_report_vgg16_v2.csv')

INFERENCE_REPS = 100
WARMUP_REPS = 20

torch.manual_seed(42)
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True



def _fallback_create_vgg16(num_classes: int = 10) -> nn.Module:
    from torchvision import models
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model.to(DEVICE)


def _fallback_build_loaders():
    from torchvision import datasets, transforms
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_TEST = 512
    NUM_WORKERS = 8

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
    tr_ld = DataLoader(tr_ds, BATCH_SIZE_TRAIN, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=(NUM_WORKERS > 0),
                       prefetch_factor=4 if NUM_WORKERS > 0 else None)
    te_ld = DataLoader(te_ds, BATCH_SIZE_TEST, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=(NUM_WORKERS > 0))
    return tr_ld, te_ld



create_vgg16 = None
build_loaders = None
for mod in ('prune_quan', 'run_mixed_strategy'):
    try:
        m = __import__(mod, fromlist=['create_vgg16', 'build_loaders'])
        create_vgg16 = getattr(m, 'create_vgg16', None)
        build_loaders = getattr(m, 'build_loaders', None)
        if create_vgg16 and build_loaders:
            break
    except Exception:
        pass

if not create_vgg16:
    create_vgg16 = _fallback_create_vgg16
if not build_loaders:
    build_loaders = _fallback_build_loaders


def get_model_size_mb(model_path: Path) -> float:
    if not model_path.exists():
        return -1.0
    size_bytes = model_path.stat().st_size
    return size_bytes / (1024 * 1024)


@torch.inference_mode()
def measure_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = correct = 0
    for x, y in tqdm(loader, desc="Measuring Accuracy", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100. * correct / total


@torch.inference_mode()
def measure_inference_latency(model: nn.Module) -> float:
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    for _ in range(WARMUP_REPS):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(INFERENCE_REPS):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
    avg_latency_ms = sum(timings) / len(timings)

    return avg_latency_ms


def measure_one_epoch_train_time(model: nn.Module, loader: DataLoader) -> float:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for x, y in tqdm(loader, desc="  Measuring Train Time", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def main():
    tr_loader, te_loader = build_loaders()

    models_to_test = [
        {"name": "Baseline", "path": BASELINE_MODEL_PATH},
        {"name": "Compressed", "path": COMPRESSED_MODEL_PATH},
    ]

    results = []

    for info in models_to_test:
        model_name = info["name"]
        model_path = info["path"]
        print(f"\n--- Processing Model: {model_name} ---")

        model = create_vgg16().to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)

        size_mb = get_model_size_mb(model_path)
        acc = measure_accuracy(model, te_loader)
        latency = measure_inference_latency(model)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        train_sec = measure_one_epoch_train_time(model, tr_loader)

        results.append({
            "Model": model_name,
            "Model Size (MB)": f"{size_mb:.2f}",
            "Accuracy (%)": f"{acc:.2f}",
            "Inference Latency (ms/image)": f"{latency:.3f}",
            "One Epoch Train Time (s)": f"{train_sec:.2f}",
        })

        print(f"Model Size: {size_mb:.2f} MB")
        print(f"Accuracy: {acc:.2f} %")
        print(f"Inference Latency: {latency:.3f} ms/image")
        print(f"One Epoch Train Time: {train_sec:.2f} s")

    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving comparison report to: {REPORT_CSV_PATH}")
    header = results[0].keys()
    with open(REPORT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print("Comparison report generated successfully!")
    print("--- Report Content ---")
    with open(REPORT_CSV_PATH, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == '__main__':
    main()

