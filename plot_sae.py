"""Plotting helpers for SAE feature visualizations."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

sys.path.insert(0, "/home/akaur64/SAEs/natural_image_saes")
from bigearthnet_loader import RGB_BANDS, reflectance_to_rgb  # noqa: E402

TOP_K_IMAGES = 3
HEATMAP_CMAP = "magma"
HEATMAP_VMAX_PERCENTILE = 99.0

# Center crop: 256x256 image -> 64x64 display; 64x64 embed grid -> 16x16 SAE input.
DEFAULT_IMAGE_CROP_SIZE = 64
DEFAULT_EMBED_CROP_SIZE = 16

CASHEW_CLASS_NAMES = {
    0: "No data",
    1: "Well-managed plantation",
    2: "Poorly-managed plantation",
    3: "Non-plantation",
    4: "Residential",
    5: "Background",
    6: "Uncertain",
}
CASHEW_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (34, 139, 34),
    2: (154, 205, 50),
    3: (0, 128, 0),
    4: (220, 20, 60),
    5: (211, 211, 211),
    6: (255, 215, 0),
}

SA_CROP_CLASS_NAMES = {
    0: "No data",
    1: "Lucerne/Medics",
    2: "Planted pastures",
    3: "Fallow",
    4: "Wine grapes",
    5: "Weeds",
    6: "Small grain grazing",
    7: "Wheat",
    8: "Canola",
    9: "Rooibos",
}
SA_CROP_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 211, 0),
    2: (255, 37, 37),
    3: (0, 168, 226),
    4: (255, 158, 9),
    5: (37, 111, 0),
    6: (255, 255, 0),
    7: (222, 166, 9),
    8: (111, 166, 0),
    9: (0, 175, 73),
}


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop spatial dims: (H, W, ...) -> (crop_size, crop_size, ...)."""
    h, w = arr.shape[:2]
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"cannot center-crop array {h}x{w} to {crop_size}x{crop_size}"
        )
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return arr[y0 : y0 + crop_size, x0 : x0 + crop_size]


def center_crop_spatial_embeddings(
    embeddings: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """Center-crop batch (N, H, W, D) to (N, crop_size, crop_size, D)."""
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim != 4:
        raise ValueError(
            f"expected (N, H, W, D) or (N, D), got {tuple(embeddings.shape)}"
        )
    h, w = embeddings.shape[1:3]
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"cannot center-crop embeddings {h}x{w} to {crop_size}x{crop_size}"
        )
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return embeddings[:, y0 : y0 + crop_size, x0 : x0 + crop_size]


def segmentation_palette(
    dataset_name: str | None,
) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    key = (dataset_name or "").replace("_", "-").lower()
    if "cashew" in key:
        return CASHEW_CLASS_NAMES, CASHEW_CLASS_COLORS
    if "sa-crop" in key or "sa_crop" in key:
        return SA_CROP_CLASS_NAMES, SA_CROP_CLASS_COLORS
    return {}, {}


def sample_metadata(sample, idx: int, dataset_name: str | None) -> dict:
    """Extract location / time metadata for plot titles."""
    name = sample.sample_name
    key = (dataset_name or "").replace("_", "-").lower()
    location_id: str | int | None = None
    timestep: str | int | None = None
    short_id: str | None = None

    if "cashew" in key:
        sample_num = int(name.split("_")[1])
        location_id = sample_num - (idx % 25)
        timestep = idx % 25
        short_id = f"loc_{location_id}_{timestep}"
    elif "sa-crop" in key or "sa_crop" in key:
        # name: ..._train_labels_{field_id}_{timestep}; last token = time index 0-4
        location_id, timestep = name.rsplit("_", 2)[-2:]
        short_id = f"loc_{location_id}_{timestep}"
    else:
        short_id = name

    return {
        "sample_name": name,
        "location_id": location_id,
        "timestep": timestep,
        "short_id": short_id,
    }


def sample_title(
    activation: float | str,
    role: str,
    meta: dict,
    idx: int,
) -> str:
    line1 = f"{activation:.3f}" if isinstance(activation, float) else str(activation)
    bits = [role]
    if meta.get("short_id"):
        bits.append(meta["short_id"])
    else:
        if meta.get("location_id") is not None:
            bits.append(f"loc {meta['location_id']}")
        if meta.get("timestep") is not None:
            bits.append(f"t{meta['timestep']}")
    bits.append(f"idx {idx}")
    return f"{line1}\n{' · '.join(bits)}" if line1 else " · ".join(bits)


def extract_segmentation_mask(sample) -> np.ndarray | None:
    label = sample.label
    if hasattr(label, "data"):
        mask = np.asarray(label.data)
        if mask.ndim == 2:
            return mask
    return None


def mask_from_label_tensor(
    label_masks: torch.Tensor | np.ndarray | None,
    idx: int,
) -> np.ndarray | None:
    if label_masks is None:
        return None
    mask = label_masks[idx]
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    return mask if mask.ndim == 2 else None


def colorize_segmentation(
    mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """Map integer mask to an (H, W, 3) RGB image in [0, 1]."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for class_id, color in class_colors.items():
        rgb[mask == class_id] = np.array(color, dtype=np.float32) / 255.0
    unknown = ~np.isin(mask, list(class_colors.keys()))
    if unknown.any():
        rgb[unknown] = (0.5, 0.5, 0.5)
    return rgb


def load_sample_from_dataset(
    dataset,
    idx: int,
    *,
    do_center_crop: bool = False,
    image_crop_size: int = DEFAULT_IMAGE_CROP_SIZE,
    label_masks: torch.Tensor | np.ndarray | None = None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    sample = dataset[idx]
    meta = sample_metadata(sample, idx, dataset_name)
    rgb = reflectance_to_rgb(
        sample.pack_to_3d(band_names=RGB_BANDS)[0],
        percentile_stretch=True,
        p_low=2.0,
        p_high=98.0,
    )
    mask = mask_from_label_tensor(label_masks, idx)
    if mask is None:
        mask = extract_segmentation_mask(sample)
    if do_center_crop:
        rgb = center_crop(rgb, image_crop_size)
        if mask is not None:
            mask = center_crop(mask, image_crop_size)
    return rgb, mask, meta


@torch.no_grad()
def spatial_feature_heatmap(
    embedding: torch.Tensor,
    sae,
    feature_idx: int,
) -> np.ndarray | None:
    """Encode embedding and return (H, W) map for feature_idx.

    Returns None for CLS embeddings (1D) where no spatial map exists.
    """
    if embedding.ndim == 1:
        return None
    if embedding.ndim != 3:
        raise ValueError(
            f"expected embedding shape (H, W, D) or (D,), got {tuple(embedding.shape)}"
        )
    sae.eval()
    _, feature_acts = sae.encode(embedding)
    return feature_acts[..., int(feature_idx)].detach().float().cpu().numpy()


def upsample_heatmap(
    heatmap: np.ndarray,
    out_hw: tuple[int, int],
    *,
    mode: str = "nearest",
) -> np.ndarray:
    """Upsample (H, W) heatmap to (out_h, out_w).

    Default ``nearest`` keeps SAE patch cells as sharp blocks (e.g. 16x16 ->
    64x64 yields clean 4x4 pixel cells). Bilinear creates cross/star artifacts
    that look like a fake non-aligned grid.
    """
    h, w = out_hw
    t = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    up = torch.nn.functional.interpolate(t, size=(h, w), mode=mode)
    return up.squeeze(0).squeeze(0).numpy()


def _row_color_scale(
    heatmaps: list[np.ndarray | None],
    percentile: float = HEATMAP_VMAX_PERCENTILE,
) -> float:
    """Shared vmax for a feature row (percentile over all row heatmaps)."""
    vals = np.concatenate([h.ravel() for h in heatmaps if h is not None], axis=0)
    if vals.size == 0:
        return 1.0
    vmax = float(np.percentile(vals, percentile))
    if vmax <= 0:
        vmax = float(vals.max()) if vals.size else 1.0
    return max(vmax, 1e-8)


def show_heatmap(
    ax,
    heatmap: np.ndarray | None,
    out_hw: tuple[int, int],
    *,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_CMAP,
) -> None:
    """Draw SAE spatial map as its own panel (nearest = sharp 4x4 patch blocks)."""
    if heatmap is None:
        ax.axis("off")
        return
    up = upsample_heatmap(heatmap, out_hw, mode="nearest")
    ax.imshow(up, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cell_h = out_hw[0] / heatmap.shape[0]
    cell_w = out_hw[1] / heatmap.shape[1]
    for i in range(1, heatmap.shape[0]):
        ax.axhline(i * cell_h - 0.5, color="w", lw=0.3, alpha=0.35)
    for j in range(1, heatmap.shape[1]):
        ax.axvline(j * cell_w - 0.5, color="w", lw=0.3, alpha=0.35)
    ax.axis("off")


def _add_segmentation_legend(
    fig,
    class_ids: set[int],
    class_names: dict[int, str],
    class_colors: dict[int, tuple[int, int, int]],
) -> None:
    handles = []
    for class_id in sorted(class_ids):
        color = class_colors.get(class_id, (128, 128, 128))
        name = class_names.get(class_id, f"class {class_id}")
        handles.append(
            Patch(
                facecolor=np.array(color, dtype=np.float32) / 255.0,
                edgecolor="0.3",
                label=name,
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 5),
            fontsize=7,
            frameon=True,
            title="Segmentation classes",
            title_fontsize=8,
        )


def plot_input_features(
    input_idx: int,
    feature_ids: list[int],
    features_examples: dict,
    input_acts: dict[int, float],
    input_dataset,
    examples_dataset,
    out_dir: str,
    *,
    sae=None,
    input_embedding: torch.Tensor | None = None,
    example_embeddings: torch.Tensor | None = None,
    show_heatmaps: bool = True,
    show_segmentation: bool = True,
    dataset_name: str | None = None,
    input_label_masks: torch.Tensor | np.ndarray | None = None,
    example_label_masks: torch.Tensor | np.ndarray | None = None,
    center_crop: bool = False,
    image_crop_size: int = DEFAULT_IMAGE_CROP_SIZE,
):
    """One figure per input image: each row is a top feature.

    Each slot shows image | segmentation map | heatmap (when available).
    """
    input_acts = input_acts or {}
    use_heatmaps = (
        show_heatmaps
        and sae is not None
        and input_embedding is not None
        and example_embeddings is not None
        and input_embedding.ndim == 3
    )

    seg_names, seg_colors = segmentation_palette(dataset_name)
    present_class_ids: set[int] = set()

    input_rgb, input_mask, input_meta = load_sample_from_dataset(
        input_dataset,
        input_idx,
        do_center_crop=center_crop,
        image_crop_size=image_crop_size,
        label_masks=input_label_masks,
        dataset_name=dataset_name,
    )
    if input_mask is not None:
        present_class_ids.update(int(v) for v in np.unique(input_mask))

    has_segmentation = show_segmentation and input_mask is not None
    n_rows = len(feature_ids)
    n_slots = TOP_K_IMAGES + 1
    if use_heatmaps and has_segmentation:
        panels_per_slot = 3
    elif use_heatmaps or has_segmentation:
        panels_per_slot = 2
    else:
        panels_per_slot = 1
    n_cols = n_slots * panels_per_slot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.0, n_rows * 1.9))
    axes = np.atleast_2d(axes)

    def slot_panel_cols(slot_idx: int) -> tuple[int, int | None, int | None]:
        base = slot_idx * panels_per_slot
        if panels_per_slot == 3:
            return base, base + 1, base + 2
        if panels_per_slot == 2:
            if use_heatmaps:
                return base, None, base + 1
            return base, base + 1, None
        return base, None, None

    def draw_slot(
        row: int,
        slot_idx: int,
        rgb: np.ndarray,
        mask: np.ndarray | None,
        heatmap: np.ndarray | None,
        title: str,
        *,
        row_label: str | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        img_col, seg_col, heat_col = slot_panel_cols(slot_idx)
        ax = axes[row, img_col]
        ax.imshow(rgb)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
        if row_label is not None:
            ax.text(
                -0.08,
                0.5,
                row_label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9,
            )

        if seg_col is not None:
            seg_ax = axes[row, seg_col]
            if mask is not None and has_segmentation:
                present_class_ids.update(int(v) for v in np.unique(mask))
                seg_ax.imshow(colorize_segmentation(mask, seg_colors))
            seg_ax.axis("off")

        if heat_col is not None and use_heatmaps:
            show_heatmap(
                axes[row, heat_col],
                heatmap,
                rgb.shape[:2],
                vmin=vmin,
                vmax=vmax,
            )

    for row, feature_id in enumerate(feature_ids):
        feat = int(feature_id)
        example = features_examples[feat]
        img_idxs = example["img_idxs"]
        acts = example["acts"]
        input_act = float(input_acts.get(feat, float("nan")))

        input_heat = None
        example_heats: list[np.ndarray | None] = []
        if use_heatmaps:
            input_heat = spatial_feature_heatmap(input_embedding, sae, feat)
            for pair_idx in range(min(len(img_idxs), TOP_K_IMAGES)):
                example_heats.append(
                    spatial_feature_heatmap(
                        example_embeddings[img_idxs[pair_idx]], sae, feat
                    )
                )
            vmax = _row_color_scale([input_heat, *example_heats])
            vmin = 0.0
        else:
            vmax, vmin = 1.0, 0.0

        draw_slot(
            row,
            0,
            input_rgb,
            input_mask,
            input_heat,
            sample_title(input_act, "input", input_meta, input_idx),
            row_label=f"feature {feat}",
            vmin=vmin,
            vmax=vmax,
        )

        for slot_idx in range(1, n_slots):
            pair_idx = slot_idx - 1
            if pair_idx >= len(img_idxs):
                for col in slot_panel_cols(slot_idx):
                    if col is not None:
                        axes[row, col].axis("off")
                continue
            ex_idx = img_idxs[pair_idx]
            ex_rgb, ex_mask, ex_meta = load_sample_from_dataset(
                examples_dataset,
                ex_idx,
                do_center_crop=center_crop,
                image_crop_size=image_crop_size,
                label_masks=example_label_masks,
                dataset_name=dataset_name,
            )
            draw_slot(
                row,
                slot_idx,
                ex_rgb,
                ex_mask,
                example_heats[pair_idx] if use_heatmaps else None,
                sample_title(acts[pair_idx], "ex", ex_meta, ex_idx),
                vmin=vmin,
                vmax=vmax,
            )

    if has_segmentation and present_class_ids:
        _add_segmentation_legend(fig, present_class_ids, seg_names, seg_colors)

    fig.suptitle(sample_title("", "scene", input_meta, input_idx), fontsize=12)
    bottom = 0.14 if has_segmentation and present_class_ids else 0.06
    fig.subplots_adjust(left=0.14, top=0.90, bottom=bottom, wspace=0.05, hspace=0.45)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{input_meta.get('short_id', input_meta['sample_name'])}_{input_idx}_features.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {out_path}")
