"""
Focal loss vs cross-entropy on CIFAR-10 with a small CNN — Modal runner.

Run (from the project root, Claude shells out):
  py -3.13 -m modal run research/focal-loss-cifar10/experiments/modal/modal_app.py \
      --config-json '{"exp_id":"exp00","loss":"ce","split":"balanced","smoke":true}'

The local entrypoint prints  RESULT_JSON:{...}  which the controller parses.
Metrics + progress are also persisted to the Modal Volume under <exp_id>/.
"""
import json
import os
import time

import modal

SLUG = "focal-loss-cifar10"
APP_NAME = f"research-{SLUG}"
VOL_NAME = f"research-{SLUG}"
DATA_ROOT = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "numpy",
    )
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@app.function(image=image, volumes={DATA_ROOT: vol}, gpu="A100", timeout=2 * 60 * 60)
def train(config: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as T

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_id = config.get("exp_id", "exp")
    loss_name = config.get("loss", "ce")          # "ce" or "focal"
    gamma = float(config.get("gamma", 2.0))
    split = config.get("split", "balanced")        # "balanced" or "lt100"
    imb_factor = float(config.get("imb_factor", 100.0))
    smoke = bool(config.get("smoke", False))
    epochs = 1 if smoke else int(config.get("epochs", 30))
    batch_size = int(config.get("batch_size", 128))

    run_dir = os.path.join(DATA_ROOT, exp_id)
    os.makedirs(run_dir, exist_ok=True)
    progress_path = os.path.join(run_dir, "progress.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    curve_path = os.path.join(run_dir, "curve.csv")

    # ---- data -------------------------------------------------------------
    data_dir = os.path.join(DATA_ROOT, "datasets")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    vol.commit()  # cache the downloaded dataset in the Volume

    targets = np.array(train_set.targets)
    num_classes = 10

    # long-tailed subsampling (exponential profile); test set stays balanced
    if split == "lt100":
        n_max = 5000
        keep_idx = []
        for c in range(num_classes):
            n_c = int(round(n_max * (1.0 / imb_factor) ** (c / (num_classes - 1))))
            cls_idx = np.where(targets == c)[0]
            rng = np.random.RandomState(c)
            keep_idx.extend(rng.choice(cls_idx, size=min(n_c, len(cls_idx)), replace=False).tolist())
        keep_idx = sorted(keep_idx)
        train_set = Subset(train_set, keep_idx)
        class_hist = [int((targets[keep_idx] == c).sum()) for c in range(num_classes)]
    else:
        class_hist = [int((targets == c).sum()) for c in range(num_classes)]

    if smoke:  # tiny subset for a fast pipeline check
        base = train_set
        idx = list(range(min(2000, len(base))))
        train_set = Subset(base, idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # ---- model: small CNN -------------------------------------------------
    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def loss_fn(logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        if loss_name == "focal":
            pt = torch.exp(-ce).clamp(1e-7, 1.0)
            return ((1 - pt) ** gamma * ce).mean()
        return ce.mean()

    # ---- evaluation (accuracy, ECE, macro-F1, worst-class) ----------------
    @torch.no_grad()
    def evaluate():
        model.eval()
        all_conf, all_correct = [], []
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
            all_conf.append(conf.cpu().numpy())
            all_correct.append((pred == y).cpu().numpy())
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[t, p] += 1
        conf = np.concatenate(all_conf)
        correct = np.concatenate(all_correct).astype(np.float64)
        acc = float(correct.mean())
        # ECE, 15 bins
        bins = np.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() > 0:
                ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
        per_class_acc = [float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(num_classes)]
        # macro-F1
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        return {
            "test_acc": acc, "ece": float(ece), "macro_f1": float(np.mean(f1s)),
            "worst_class_acc": float(min(per_class_acc)), "per_class_acc": per_class_acc,
        }

    # ---- train loop -------------------------------------------------------
    with open(curve_path, "w") as f:
        f.write("epoch,train_loss,test_acc,ece\n")
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        train_loss = running / max(len(train_loader), 1)
        ev = evaluate()
        best_acc = max(best_acc, ev["test_acc"])
        with open(curve_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{ev['test_acc']:.4f},{ev['ece']:.4f}\n")
        _write_json(progress_path, {
            "exp_id": exp_id, "epoch": epoch + 1, "total_epochs": epochs,
            "train_loss": round(train_loss, 4), "test_acc": round(ev["test_acc"], 4),
            "best_acc": round(best_acc, 4), "done": False,
        })
        vol.commit()

    final = evaluate()
    metrics = {
        "exp_id": exp_id, "loss": loss_name, "gamma": gamma if loss_name == "focal" else None,
        "split": split, "epochs": epochs, "n_params": int(n_params),
        "test_acc": round(final["test_acc"], 4), "best_acc": round(best_acc, 4),
        "ece": round(final["ece"], 4), "macro_f1": round(final["macro_f1"], 4),
        "worst_class_acc": round(final["worst_class_acc"], 4),
        "per_class_acc": [round(a, 4) for a in final["per_class_acc"]],
        "class_hist": class_hist, "train_minutes": round((time.time() - t0) / 60, 2),
        "smoke": smoke, "device": device,
    }
    _write_json(metrics_path, metrics)
    _write_json(progress_path, {**{k: metrics[k] for k in ("exp_id", "test_acc", "ece")},
                                "epoch": epochs, "total_epochs": epochs, "done": True})
    vol.commit()
    return metrics


@app.local_entrypoint()
def main(config_json: str = "{}"):
    # Accept "@path/to/config.json" to avoid shell quoting issues (esp. Windows PowerShell).
    if config_json.startswith("@"):
        with open(config_json[1:], "r", encoding="utf-8") as f:
            config_json = f.read()
    config = json.loads(config_json)
    gpu = config.get("gpu", "A100")
    fn = train.with_options(gpu=gpu) if gpu != "A100" else train
    result = fn.remote(config)
    print("RESULT_JSON:" + json.dumps(result))
