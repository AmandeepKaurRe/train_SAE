"""Embedding loaders: in-RAM datasets and disk-backed memmaps for SAE training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MemmapRef:
    path: Path
    n_rows: int
    d_in: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.d_in


class EmbeddingDataset(Dataset):
    """In-RAM float32 rows (CLS / --no-memmap)."""

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings.contiguous()

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


class MemmapBatchDataset(Dataset):
    """Each item is one contiguous memmap slice; only batch *order* is shuffled."""

    def __init__(
        self,
        path: Path,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.path = Path(path)
        self._arr = np.load(self.path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"expected 2D memmap, got {self._arr.shape} at {self.path}")
        if self._arr.dtype != np.float32:
            raise ValueError(f"expected float32 memmap, got {self._arr.dtype} at {self.path}")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        n_rows = int(self._arr.shape[0])
        if drop_last:
            self._num_batches = n_rows // batch_size
            self._n_rows = self._num_batches * batch_size
        else:
            self._num_batches = (n_rows + batch_size - 1) // batch_size
            self._n_rows = n_rows

        self._batch_order = np.arange(self._num_batches, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._arr.shape[0]), int(self._arr.shape[1])

    def set_epoch(self, epoch: int) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            order = np.arange(self._num_batches, dtype=np.int64)
            rng.shuffle(order)
            self._batch_order = order
        else:
            self._batch_order = np.arange(self._num_batches, dtype=np.int64)

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> torch.Tensor:
        batch_id = int(self._batch_order[idx])
        start = batch_id * self.batch_size
        stop = min(start + self.batch_size, self._n_rows)
        return torch.from_numpy(np.asarray(self._arr[start:stop], dtype=np.float32).copy())


def make_memmap_dataloader(
    dataset: MemmapBatchDataset,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    loader_workers = max(num_workers, 0)
    kwargs: dict = {
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": lambda batch: batch[0],
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
        # Workers fork a copy of the dataset; disable persistence so set_epoch()
        # on the main process is picked up when a new iterator starts each epoch.
        "persistent_workers": False,
    }
    if loader_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def load_embeddings(path: str, is_seg: bool) -> torch.Tensor:
    """Load embeddings; drop labels immediately to avoid holding mask tensors in RAM."""
    p = Path(path)
    data = torch.load(p, map_location="cpu", weights_only=False)
    emb = data["embeddings"]
    del data
    if is_seg:
        if emb.ndim <= 2:
            raise ValueError(f"{p}: expected spatial embeddings for --seg, got {tuple(emb.shape)}")
        emb = emb.reshape(-1, emb.shape[-1])
    out = emb.float().contiguous()
    del emb
    return out


def collect_split_paths(
    root: str,
    model: str,
    datasets: list[str],
    split: str,
    *,
    seg: bool,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return existing paths and flat (n_rows, d_in) by loading each file once for shape."""
    paths: list[str] = []
    shapes: list[tuple[int, int]] = []
    for dataset in datasets:
        path = f"{root}/{model}/{dataset}/{split}.pt"
        if not Path(path).exists():
            print(f"skip missing {path}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        emb = data["embeddings"]
        del data
        spatial = emb.ndim > 2
        if seg and not spatial:
            del emb
            continue
        if not seg and spatial:
            del emb
            continue
        if seg:
            n, d = emb.shape[0] * emb.shape[1] * emb.shape[2], int(emb.shape[-1])
        else:
            n, d = int(emb.shape[0]), int(emb.shape[1])
        del emb
        paths.append(path)
        shapes.append((n, d))
    return paths, shapes


def accumulate_embeddings(
    paths: list[str],
    shapes: list[tuple[int, int]],
    *,
    is_seg: bool,
) -> torch.Tensor:
    """Load datasets one-by-one into a single preallocated tensor."""
    n_total = sum(n for n, _ in shapes)
    d_in = shapes[0][1]
    print(
        f"  preallocating ({n_total}, {d_in}) float32 "
        f"({n_total * d_in * 4 / 1e9:.2f} GB)"
    )
    out = torch.empty((n_total, d_in), dtype=torch.float32)
    offset = 0
    for path, (n, _) in zip(paths, shapes):
        emb = load_embeddings(path, is_seg=is_seg)
        if emb.shape[0] != n or emb.shape[1] != d_in:
            raise RuntimeError(
                f"shape mismatch for {path}: got {tuple(emb.shape)}, expected {(n, d_in)}"
            )
        out[offset : offset + n].copy_(emb)
        offset += n
        del emb
        print(f"  loaded {path} -> {n} rows ({offset}/{n_total})")

    return out


def _memmap_meta_path(npy_path: Path) -> Path:
    return npy_path.with_suffix(npy_path.suffix + ".meta.json")


def _meta_matches(
    meta_path: Path,
    *,
    paths: list[str],
    shapes: list[tuple[int, int]],
    is_seg: bool,
    model: str,
    split: str,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    expected = {
        "paths": paths,
        "shapes": [list(s) for s in shapes],
        "is_seg": is_seg,
        "model": model,
        "split": split,
        "dtype": "float32",
    }
    return (
        meta.get("paths") == expected["paths"]
        and meta.get("shapes") == expected["shapes"]
        and meta.get("is_seg") == expected["is_seg"]
        and meta.get("model") == expected["model"]
        and meta.get("split") == expected["split"]
        and meta.get("dtype") == expected["dtype"]
    )


def build_or_load_memmap(
    memmap_dir: Path,
    split: str,
    paths: list[str],
    shapes: list[tuple[int, int]],
    *,
    is_seg: bool,
    model: str,
) -> MemmapRef:
    """Build a float32 .npy once (streaming from .pt files), then reopen as mmap."""
    memmap_dir = Path(memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    npy_path = memmap_dir / f"{split}.npy"
    meta_path = _memmap_meta_path(npy_path)

    n_total = sum(n for n, _ in shapes)
    d_in = shapes[0][1]
    nbytes = n_total * d_in * 4

    if npy_path.exists() and _meta_matches(
        meta_path, paths=paths, shapes=shapes, is_seg=is_seg, model=model, split=split
    ):
        print(f"  reusing memmap {npy_path} ({n_total}, {d_in}) [{nbytes / 1e9:.2f} GB on disk]")
        return MemmapRef(npy_path, n_total, d_in)

    print(
        f"  writing memmap {npy_path} ({n_total}, {d_in}) float32 "
        f"({nbytes / 1e9:.2f} GB on disk)..."
    )
    arr = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(n_total, d_in)
    )
    offset = 0
    try:
        for path, (n, _) in zip(paths, shapes):
            data = torch.load(Path(path), map_location="cpu", weights_only=False)
            emb = data["embeddings"]
            del data

            if is_seg:
                if emb.ndim <= 2:
                    raise ValueError(
                        f"{path}: expected spatial embeddings for --seg, got {tuple(emb.shape)}"
                    )
                emb = emb.reshape(-1, emb.shape[-1])

            if emb.shape[0] != n or emb.shape[1] != d_in:
                raise RuntimeError(
                    f"shape mismatch for {path}: got {tuple(emb.shape)}, expected {(n, d_in)}"
                )

            target_chunk_bytes = 128 * 1024**2
            chunk_rows = max(1024, int(target_chunk_bytes // (d_in * 4)))

            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                chunk = emb[start:stop]
                if chunk.dtype != torch.float32:
                    chunk_f32 = chunk.to(torch.float32)
                else:
                    chunk_f32 = chunk if chunk.is_contiguous() else chunk.contiguous()
                arr[offset + start : offset + stop] = chunk_f32.numpy()
                del chunk_f32, chunk

            offset += n
            del emb
            arr.flush()
            print(f"  wrote {path} -> {n} rows ({offset}/{n_total})")
    except Exception:
        del arr
        if npy_path.exists():
            npy_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        raise

    del arr
    meta = {
        "paths": paths,
        "shapes": [list(s) for s in shapes],
        "is_seg": is_seg,
        "model": model,
        "split": split,
        "dtype": "float32",
        "shape": [n_total, d_in],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  memmap ready: {npy_path}")
    return MemmapRef(npy_path, n_total, d_in)
