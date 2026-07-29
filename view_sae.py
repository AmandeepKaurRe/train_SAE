"""
View the SAE model and its latent space features using top activating images.
"""
from __future__ import annotations

import os
import sys

import torch
from tqdm import tqdm
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.sae.sae import StandardSparseAutoencoder
import numpy as np
import geobench

# Local helpers live next to this script (not the old /home/akaur64/sae tree).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_sae import (
    TOP_K_IMAGES,
    DEFAULT_EMBED_CROP_SIZE,
    DEFAULT_IMAGE_CROP_SIZE,
    center_crop_spatial_embeddings,
    plot_input_features,
)

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# All-patches Galileo SAE (trained with train_sae.py --seg on spatial patch rows).
# Older CLS-only checkpoint (if needed): .../galileo_tiny/m_eurosat/prisma_sae.pt
GALILEO_PATH = "/scratch/akaur64/olmo-embeddings/galileo_tiny/prisma_sae.pt"
GALILEO_EMBEDDINGS_PATH = "/scratch/akaur64/olmo-embeddings/galileo_tiny"
TOP_K_FEATURES = 5
DEVICE = torch.device("cpu")
IN_DIR = "/home/akaur64/SAEs/train_SAE/Eurosat_images"

# Spatial / all-patches datasets (embeddings are [N, H, W, D]).
DATASET = "m_sa_crop_type"
# DATASET = "m_cashew_plant"
# DATASET = "m_eurosat"  # CLS [N, D] — not all-patches
TASK_TYPE = (
    "segmentation_v1.0"
    if DATASET in ["m_cashew_plant", "m_sa_crop_type"]
    else "classification_v1.0"
)
IN_SPLIT = "test"
OUT_SPLIT = "train"
OUT_DIR = "/home/akaur64/SAEs/train_SAE/out_features"

# Center-crop: 256x256 RGB -> 64x64; 64x64 embed -> 16x16 (each SAE cell = 4x4 RGB px).
# Heatmaps use nearest upsample so those 4x4 patch blocks stay sharp (bilinear made
# cross/star artifacts that looked like a fake grid). Set False for full 64x64 maps.
USE_CENTER_CROP = True
CENTER_CROP_IMAGE_SIZE = DEFAULT_IMAGE_CROP_SIZE
CENTER_CROP_EMBED_SIZE = DEFAULT_EMBED_CROP_SIZE

DATASETS = [
    "awf_sentinel1",
    "awf_sentinel2",
    "breizhcrops",
    "cropharvest_Peoples_Republic_of_China_6",
    "cropharvest_Peoples_Republic_of_China_6_sentinel1",
    "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
    "cropharvest_Togo_12_sentinel1",
    "cropharvest_Togo_12_sentinel2",
    "cropharvest_Togo_12_sentinel2_sentinel1",
    "m_bigearthnet",
    "m_brick_kiln",
    "m_cashew_plant",
    "m_eurosat",
    "m_sa_crop_type",
    "m_so2sat",
    "mados",
    "nandi_sentinel1",
    "nandi_sentinel2",
    "pastis_sentinel1",
    "pastis_sentinel1_sentinel2",
    "pastis_sentinel2",
    "sen1floods11",
]


def load_sae(path: str, device: torch.device = DEVICE) -> SparseAutoencoder:
    """Load SAE onto CPU even if the checkpoint was saved with device=cuda."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    cfg.device = str(device)
    sae = StandardSparseAutoencoder(cfg=cfg)
    sae.load_state_dict(state["state_dict"])
    return sae.to(device)


def load_embeddings(dataset: str, split: str):
    if dataset == "all":
        embeddings = []
        labels = []
        for ds_name in tqdm(DATASETS, total=len(DATASETS), desc="Loading embeddings"):
            data = torch.load(
                f"{GALILEO_EMBEDDINGS_PATH}/{ds_name}/train.pt",
                map_location="cpu",
                weights_only=False,
            )
            emb = data["embeddings"].float()
            if emb.ndim > 2:
                continue
            embeddings.append(emb)
            labels.append(data["labels"])
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
    data = torch.load(
        f"{GALILEO_EMBEDDINGS_PATH}/{dataset}/{split}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return data["embeddings"].float(), data["labels"]

def load_image_embeddings(embeddings, idx) -> torch.Tensor:
    return embeddings[idx]


def maybe_center_crop_embeddings(
    embeddings: torch.Tensor,
    *,
    enabled: bool,
    embed_crop_size: int,
) -> torch.Tensor:
    if not enabled or embeddings.ndim != 4:
        return embeddings
    cropped = center_crop_spatial_embeddings(embeddings, embed_crop_size)
    print(
        f"center-cropped embeddings {tuple(embeddings.shape)} "
        f"-> {tuple(cropped.shape)}"
    )
    return cropped


def assert_mask_order(
    label_masks: torch.Tensor,
    dataset,
    name: str,
    *,
    max_check: int | None = 50,
) -> None:
    """Fail if embedding label masks[i] != geobench sample mask at i."""
    assert len(dataset) == label_masks.shape[0], (
        f"{name}: dataset len {len(dataset)} != masks {label_masks.shape[0]}"
    )
    n = len(dataset) if max_check is None else min(len(dataset), max_check)
    mismatches = []
    for i in range(n):
        gb_mask = np.asarray(dataset[i].label.data)
        pt_mask = label_masks[i].detach().cpu().numpy()
        if gb_mask.shape != pt_mask.shape or not np.array_equal(gb_mask, pt_mask):
            mismatches.append(i)
    if mismatches:
        examples = ", ".join(str(i) for i in mismatches[:5])
        raise AssertionError(
            f"{name}: {len(mismatches)}/{n} mask mismatches at indices "
            f"{examples} — embedding indices may not match the image dataset"
        )
    print(f"{name}: mask order OK (checked {n} samples)")


def assert_label_order(labels: torch.Tensor, dataset, name: str) -> None:
    """Fail if embedding labels[i] != dataset[i].label (index misalignment)."""
    assert len(dataset) == labels.shape[0], (
        f"{name}: dataset len {len(dataset)} != embeddings {labels.shape[0]}"
    )
    ds_labels = torch.tensor(
        [
            int(dataset[i].label)
            for i in tqdm(range(len(dataset)), desc=f"Checking {name} label order")
        ],
        dtype=torch.long,
    )
    pt_labels = labels.detach().cpu().long().view(-1)
    mismatch = ds_labels != pt_labels
    n_mismatch = int(mismatch.sum())
    if n_mismatch:
        bad = torch.where(mismatch)[0][:10].tolist()
        examples = ", ".join(
            f"i={i} pt={int(pt_labels[i])} ds={int(ds_labels[i])}" for i in bad
        )
        raise AssertionError(
            f"{name}: {n_mismatch}/{len(dataset)} label mismatches — "
            f"embedding indices are not aligned with the image dataset "
            f"(e.g. {examples})"
        )
    print(f"{name}: label order OK ({len(dataset)} samples)")


@torch.no_grad()
def get_top_activating_features(img_embedding, sae, top_k=TOP_K_FEATURES):
    sae.eval()
    assert img_embedding.ndim == 1 or img_embedding.ndim == 3, "Input embedding must be 1D or 3D"
    _, activations = sae.encode(img_embedding)
    if activations.ndim == 3:
        activations = activations.amax(dim=(0, 1))   # shape: (dim,)
    values, indices = torch.topk(activations, k=top_k)
    return indices.tolist(), [float(v) for v in values.tolist()]

@torch.no_grad()
def get_top_activating_images(embeddings, feature_idx, sae, top_k=TOP_K_IMAGES):
    # embeddings are (n, dim) or (n, patches, patches, dim)
    sae.eval()    
    _, feature_acts = sae.encode(embeddings) # shape: (n, patches, patches, dim) or (n, dim)
    acts = feature_acts[..., int(feature_idx)] # shape: (n, patches, patches) or (n,)
    if acts.ndim == 3:
        acts = acts.amax(dim=(1, 2)) # shape: (n,)
    k = min(top_k, acts.shape[0])
    values, indices = torch.topk(acts, k=k)
    return indices.tolist(), [float(v) for v in values.tolist()]


def load_from_geobench(dataset_name: str, split: str, task_type: str):
    needle = dataset_name.replace("_", "-").lower()
    for task in geobench.task_iterator(benchmark_name=task_type):
        if needle not in task.dataset_name.lower():
            continue
        return task.get_dataset(split=split)
    raise ValueError(
        f"No GEO-Bench task matching {dataset_name!r} in benchmark {task_type!r}"
    )

def main() -> None:
    print(f"starting... Running for dataset: {DATASET} with task type: {TASK_TYPE}")
    print(f"loading all-patches Galileo SAE from {GALILEO_PATH}")
    sae = load_sae(GALILEO_PATH, DEVICE)
    sae.eval()
    print(
        f"sae loaded on {DEVICE}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, "
        f"expansion={sae.cfg.expansion_factor}"
    )

    input_embeddings, input_labels = load_embeddings(DATASET, IN_SPLIT)
    example_embeddings, example_labels = load_embeddings(DATASET, OUT_SPLIT)
    # check_label_order(example_labels, load_eurosat(split="train"))

    example_embeddings = example_embeddings.to(DEVICE)
    input_embeddings = input_embeddings.to(DEVICE)
    print(f"loaded input embeddings ({DATASET}_{IN_SPLIT}): {tuple(input_embeddings.shape)}")
    print(f"loaded example embeddings ({DATASET}_{OUT_SPLIT}): {tuple(example_embeddings.shape)}")

    input_embeddings = maybe_center_crop_embeddings(
        input_embeddings,
        enabled=USE_CENTER_CROP,
        embed_crop_size=CENTER_CROP_EMBED_SIZE,
    )
    example_embeddings = maybe_center_crop_embeddings(
        example_embeddings,
        enabled=USE_CENTER_CROP,
        embed_crop_size=CENTER_CROP_EMBED_SIZE,
    )

    input_dataset = load_from_geobench(DATASET, IN_SPLIT, TASK_TYPE)
    examples_dataset = load_from_geobench(DATASET, OUT_SPLIT, TASK_TYPE)
    if TASK_TYPE == "segmentation_v1.0":
        # assert_mask_order(input_labels, input_dataset, name=f"{DATASET}_{IN_SPLIT}")
        assert_mask_order(example_labels, examples_dataset, name=f"{DATASET}_{OUT_SPLIT}")
    elif not TASK_TYPE == "segmentation_v1.0":
        assert_label_order(example_labels, examples_dataset, name=f"{DATASET}_{OUT_SPLIT}")
    # print(f"loaded input ds n={len(input_dataset)}, examples ds n={len(examples_dataset)}")

    input_idxs = list(range(0, 7))

    per_input_features: list[tuple[int, list[int], dict[int, float]]] = []
    all_top_features: list[int] = []
    for img_idx in input_idxs:
        img_emb = load_image_embeddings(input_embeddings, img_idx)
        top_features, top_acts = get_top_activating_features(img_emb, sae)
        input_acts = {feat: act for feat, act in zip(top_features, top_acts)}
        per_input_features.append((img_idx, top_features, input_acts))
        all_top_features.extend(top_features)

    features_examples = {}
    for feat in sorted(set(all_top_features)):
        img_idxs, acts = get_top_activating_images(
            example_embeddings, feat, sae, TOP_K_IMAGES
        )
        features_examples[feat] = {"img_idxs": img_idxs, "acts": acts}

    out_suffix = f"{OUT_DIR}_{DATASET}_{OUT_SPLIT}"
    if USE_CENTER_CROP:
        out_suffix += f"_crop{CENTER_CROP_IMAGE_SIZE}"

    for img_idx, top_features, input_acts in per_input_features:
        plot_input_features(
            img_idx,
            top_features,
            features_examples,
            input_acts,
            input_dataset,
            examples_dataset,
            out_dir=out_suffix,
            sae=sae,
            input_embedding=load_image_embeddings(input_embeddings, img_idx),
            example_embeddings=example_embeddings,
            show_heatmaps=True,
            show_segmentation=TASK_TYPE == "segmentation_v1.0",
            dataset_name=DATASET,
            input_label_masks=input_labels,
            example_label_masks=example_labels,
            center_crop=USE_CENTER_CROP,
            image_crop_size=CENTER_CROP_IMAGE_SIZE,
        )


if __name__ == "__main__":
    main()
