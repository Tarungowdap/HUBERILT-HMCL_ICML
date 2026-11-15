"""Real datasets from Kaggle (CIFAR-100, MiniImageNet) -> class-incremental task streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_cifar100(data_root: Path):
    from torchvision.datasets import CIFAR100

    tr = CIFAR100(str(data_root), train=True, download=True)
    te = CIFAR100(str(data_root), train=False, download=True)
    return (
        np.asarray(tr.data, dtype=np.uint8), np.asarray(tr.targets, dtype=np.int64),
        np.asarray(te.data, dtype=np.uint8), np.asarray(te.targets, dtype=np.int64),
    )


def load_miniimagenet(data_root: Path, img_size: int):
    """Load timm/mini-imagenet from HuggingFace, resize, and cache as .npz."""
    cache = data_root / f"miniimagenet_{img_size}.npz"
    if cache.exists():
        print(f"Loading cached MiniImageNet from {cache}")
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]

    from datasets import load_dataset
    from PIL import Image

    print("Downloading timm/mini-imagenet from HuggingFace (first run only)...")
    ds = load_dataset("timm/mini-imagenet", cache_dir=str(data_root / "hf_cache"))
    label_key = "label" if "label" in ds["train"].column_names else ds["train"].column_names[-1]
    img_key = "image"

    def convert(split):
        n = len(split)
        xs = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
        ys = np.asarray(split[label_key], dtype=np.int64)
        for i, ex in enumerate(split):
            img = ex[img_key].convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            xs[i] = np.asarray(img)
            if i % 10000 == 0:
                print(f"  resized {i}/{n}")
        return xs, ys

    xtr, ytr = convert(ds["train"])
    test_split = "validation" if "validation" in ds else "test"
    xte, yte = convert(ds[test_split])
    data_root.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    print(f"Cached to {cache}")
    return xtr, ytr, xte, yte


KAGGLE_DEFAULTS = {
    "split_cifar100": "fedesoriano/cifar100",
    "split_miniimagenet": "arjunashok33/miniimagenet",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_cifar_pickles(root: Path):
    """CIFAR-100 'python version' layout: files `train` and `test` with fine_labels."""
    import pickle
    for tr in root.rglob("train"):
        te = tr.parent / "test"
        if tr.is_file() and te.is_file():
            out = []
            for f in (tr, te):
                with open(f, "rb") as fh:
                    d = pickle.load(fh, encoding="latin1")
                x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                out += [np.ascontiguousarray(x), np.asarray(d["fine_labels"], dtype=np.int64)]
            return out
    return None


def _class_dirs(d: Path):
    return [c for c in sorted(d.iterdir()) if c.is_dir() and any(
        p.suffix.lower() in IMG_EXT for p in c.iterdir())]


def _find_image_folders(root: Path):
    """Return (train_dir, test_dir|None) holding one sub-folder per class."""
    best = None
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        kids = {c.name.lower(): c for c in d.iterdir() if c.is_dir()}
        tr = kids.get("train")
        te = kids.get("test") or kids.get("val") or kids.get("validation")
        if tr is not None and te is not None and _class_dirs(tr) and _class_dirs(te):
            return tr, te
        n = len(_class_dirs(d))
        if n >= 10 and (best is None or n > best[1]):
            best = (d, n)
    return (best[0], None) if best else (None, None)


def _load_folder(class_dirs, names, img_size):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    files, labels = [], []
    for c in class_dirs:
        fs = sorted(p for p in c.iterdir() if p.suffix.lower() in IMG_EXT)
        files += fs
        labels += [names.index(c.name)] * len(fs)

    def read(p):
        return np.asarray(Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR))

    with ThreadPoolExecutor(16) as ex:
        xs = np.stack(list(ex.map(read, files, chunksize=256)))
    return xs.astype(np.uint8), np.asarray(labels, dtype=np.int64)


def load_from_kaggle(dataset: str, slug: str, data_root: Path, img_size: int, seed: int):
    """Download a Kaggle dataset with kagglehub and auto-detect its layout."""
    size = 32 if dataset == "split_cifar100" else img_size
    cache = data_root / f"kaggle_{slug.replace('/', '__')}_{size}.npz"
    if cache.exists():
        print(f"Loading cached arrays from {cache}")
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]

    import kagglehub
    print(f"Downloading Kaggle dataset {slug} ...")
    root = Path(kagglehub.dataset_download(slug))
    print(f"  files at {root}")

    arrays = _find_cifar_pickles(root) if dataset == "split_cifar100" else None
    if arrays is None:
        tr_dir, te_dir = _find_image_folders(root)
        if tr_dir is None:
            raise RuntimeError(f"Could not recognise the layout of {root}. Run `!find {root} | head -50` "
                               "and send me the output, or pass a different --kaggle-dataset.")
        if te_dir is not None:
            names = sorted(c.name for c in _class_dirs(tr_dir))
            print(f"  image folders: {tr_dir} / {te_dir} ({len(names)} classes), resizing to {size}px ...")
            xtr, ytr = _load_folder(_class_dirs(tr_dir), names, size)
            xte, yte = _load_folder([c for c in _class_dirs(te_dir) if c.name in names], names, size)
        else:
            dirs = _class_dirs(tr_dir)
            names = [c.name for c in dirs]
            print(f"  image folders: {tr_dir} ({len(names)} classes, no test split), resizing to {size}px ...")
            x, y = _load_folder(dirs, names, size)
            rng = np.random.RandomState(1234)
            te_mask = np.zeros(len(y), dtype=bool)
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                te_mask[rng.choice(idx, len(idx) // 6, replace=False)] = True
            xtr, ytr, xte, yte = x[~te_mask], y[~te_mask], x[te_mask], y[te_mask]
        arrays = [xtr, ytr, xte, yte]

    xtr, ytr, xte, yte = arrays
    data_root.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    print(f"  {len(ytr)} train / {len(yte)} test, {int(ytr.max()) + 1} classes, cached to {cache}")
    return xtr, ytr, xte, yte


def build_stream(dataset: str, data_root: Path, num_tasks: int, classes_per_task: int,
                 img_size: int, seed: int, device: torch.device,
                 source: str = "kaggle", kaggle_dataset: str | None = None) -> dict:
    if source == "kaggle":
        xtr, ytr, xte, yte = load_from_kaggle(
            dataset, kaggle_dataset or KAGGLE_DEFAULTS[dataset], data_root, img_size, seed)
    elif dataset == "split_cifar100":
        xtr, ytr, xte, yte = load_cifar100(data_root)
    elif dataset == "split_miniimagenet":
        xtr, ytr, xte, yte = load_miniimagenet(data_root, img_size)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    n_classes = int(ytr.max()) + 1
    needed = num_tasks * classes_per_task
    if needed > n_classes:
        raise ValueError(f"{num_tasks} tasks x {classes_per_task} classes > {n_classes} classes")

    order = np.random.RandomState(seed).permutation(n_classes)[:needed]

    def to_tensor(x):
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)

    stream = {}
    for t in range(num_tasks):
        cls = order[t * classes_per_task:(t + 1) * classes_per_task]
        remap = {int(c): i for i, c in enumerate(cls)}
        mtr, mte = np.isin(ytr, cls), np.isin(yte, cls)
        ltr = np.array([remap[int(y)] for y in ytr[mtr]], dtype=np.int64)
        lte = np.array([remap[int(y)] for y in yte[mte]], dtype=np.int64)
        stream[f"task_{t+1}"] = {
            "train_x": to_tensor(xtr[mtr]),
            "train_y": torch.from_numpy(ltr).to(device),
            "test_x": to_tensor(xte[mte]),
            "test_y": torch.from_numpy(lte).to(device),
            "num_classes": classes_per_task,
            "class_offset": t * classes_per_task,
        }
    print(f"{dataset}: {num_tasks} tasks x {classes_per_task} classes, "
          f"{sum(len(v['train_y']) for v in stream.values())} train / "
          f"{sum(len(v['test_y']) for v in stream.values())} test images, "
          f"image shape {tuple(stream['task_1']['train_x'].shape[1:])}")
    return stream


class GPUAugment:
    """Normalise uint8 images; random crop (reflect pad) + horizontal flip when training."""

    def __init__(self, mean, std, pad: int = 4) -> None:
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.pad = pad

    def __call__(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        x = x.float().div_(255.0)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if not train:
            return x
        n, _, h, w = x.shape
        xp = F.pad(x, (self.pad,) * 4, mode="reflect")
        i = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
        j = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
        rows = (i.view(n, 1) + torch.arange(h, device=x.device)).view(n, 1, h, 1).expand(n, 3, h, w)
        cols = (j.view(n, 1) + torch.arange(w, device=x.device)).view(n, 1, 1, w).expand(n, 3, h, w)
        b_idx = torch.arange(n, device=x.device).view(n, 1, 1, 1)
        c_idx = torch.arange(3, device=x.device).view(1, 3, 1, 1)
        x = xp[b_idx, c_idx, rows, cols]
        flip = torch.rand(n, device=x.device) < 0.5
        x[flip] = x[flip].flip(-1)
        return x
