#!/usr/bin/env python3
"""Train a ViT-Prisma SAE directly on cached embeddings.

This is the model-free version of the library's SAE workflow:
- uses the library's SparseAutoencoder implementation
- logs metrics to Weights & Biases
- saves checkpoints with the library helper
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import wandb
except Exception:
    wandb = None

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import StandardSparseAutoencoder

# MODELS = [
#     "croma_base",
#     "dino_v3_dinov3_vitb16",
#     "galileo_tiny",
#     "olmoearth_tiny",
#     # "presto",
# ]
MODEL = "galileo_tiny"
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

SPLITS = ["train", "valid", "test"]

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings.contiguous()

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


def load_embeddings(path: str, is_seg: bool, load_labels: bool = False) -> torch.Tensor:
    p = Path(path)
    data = torch.load(p, map_location="cpu", weights_only=False)
    if is_seg:
        data['embeddings'] = data['embeddings'].reshape(-1, data['embeddings'].shape[-1])
    if load_labels:
        return data["embeddings"].float(), data["labels"]
    else:
        return data["embeddings"].float()



def build_config(
    d_in: int,
    expansion_factor: int,
    device: str,
    log_to_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    checkpoint_path: str,
) -> VisionModelSAERunnerConfig:
    cfg = VisionModelSAERunnerConfig()

    # Core SAE shape / training setup
    cfg.d_in = d_in
    cfg.expansion_factor = expansion_factor
    cfg.architecture = "standard"
    cfg.activation_fn_str = "relu"
    cfg.activation_fn_kwargs = {}
    cfg.normalize_activations = "none"
    cfg.b_dec_init_method = "mean"
    cfg.initialization_method = "independent"
    cfg.l1_coefficient = 1e-4
    cfg.lp_norm = 1
    cfg.lr = 1e-3
    cfg.train_batch_size = 1024
    cfg.num_epochs = 20
    cfg.seed = 42
    cfg._device = device
    cfg._dtype = "float32"

    # These are part of the Prisma config even though we are not using the
    # hooked trainer in this no-model setup.
    cfg.use_cached_activations = True
    cfg.cached_activations_path = None
    cfg.log_to_wandb = log_to_wandb
    cfg.wandb_project = wandb_project
    cfg.wandb_entity = wandb_entity
    cfg.wandb_log_frequency = 10
    cfg.n_checkpoints = 10
    cfg.checkpoint_path = checkpoint_path
    cfg.feature_sampling_window = 1000
    cfg.dead_feature_window = 5000
    cfg.dead_feature_threshold = 1e-8
    cfg.min_l0 = None
    cfg.min_explained_variance = None

    return cfg


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1).mean()

def explained_variance(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    # used vit-prisma implementation
    per_token_l2_loss = (recon - x).pow(2).sum(dim=-1).squeeze()
    total_variance = (x - x.mean(0)).pow(2).sum(-1)
    return (1.0 - per_token_l2_loss / (total_variance + 1e-8)) # plot both mean and std

def l0_sparsity(feature_acts: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return (feature_acts.abs() > threshold).float().sum(dim=-1).mean() # higher means less sparse?

def dead_feature_fraction(feature_acts: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    # Features whose mean absolute activation is effectively zero. nxD -> 1
    mean_abs = feature_acts.abs().mean(dim=0)
    return (mean_abs <= threshold).float().mean()


def train_one_epoch(
    sae: StandardSparseAutoencoder,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: VisionModelSAERunnerConfig,
    device: torch.device,
) -> dict:
    sae.train()
    totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "cos": 0.0, "ev_mean": 0.0, "ev_std": 0.0, "l0": 0.0, "dead_frac": 0.0}
    n_batches = 0

    for x in loader:
        x = x.to(device)
        recon, feat, loss, mse, l1, ghost, aux = sae(x)
        # _, feat, hidden_pre = sae.encode(x, return_hidden_pre=True)
        # recon = sae.decode(feat)

        # mse = F.mse_loss(recon, x)
        # l1 = feat.abs().mean()
        # loss = mse + cfg.l1_coefficient * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(cfg, "max_grad_norm", None):
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.max_grad_norm)
        opt.step()
        sae.set_decoder_norm_to_unit_norm()

        totals["loss"] += loss.item()
        totals["mse"] += mse.item()
        totals["l1"] += l1.item()
        totals["cos"] += cosine_similarity(x, recon).item()
        totals["ev_mean"] += explained_variance(x, recon).mean().item()
        totals["ev_std"] += explained_variance(x, recon).std().item()
        totals["l0"] += l0_sparsity(feat).item()
        totals["dead_frac"] += dead_feature_fraction(feat, cfg.dead_feature_threshold).item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(sae: StandardSparseAutoencoder, loader: DataLoader, cfg: VisionModelSAERunnerConfig, device: torch.device) -> dict:
    sae.eval()
    totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "cos": 0.0, "ev_mean": 0.0, "ev_std": 0.0, "l0": 0.0, "dead_frac": 0.0}
    n_batches = 0

    for x in loader:
        x = x.to(device)
        recon, feat, loss, mse, l1, ghost, aux = sae(x)
        # _, feat, hidden_pre = sae.encode(x, return_hidden_pre=True)
        # recon = sae.decode(feat)

        # mse = F.mse_loss(recon, x)
        # l1 = feat.abs().mean()
        # loss = mse + cfg.l1_coefficient * l1

        totals["loss"] += loss.item()
        totals["mse"] += mse.item()
        totals["l1"] += l1.item()
        totals["cos"] += cosine_similarity(x, recon).item()
        totals["ev_mean"] += explained_variance(x, recon).mean().item()
        totals["ev_std"] += explained_variance(x, recon).std().item()
        totals["l0"] += l0_sparsity(feat).item()
        totals["dead_frac"] += dead_feature_fraction(feat, cfg.dead_feature_threshold).item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--embeddings", default='/scratch/akaur64/olmo-embeddings/galileo_tiny/m_eurosat/train.pt', help="Path to cached embeddings (.pt or .npy)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--wandb-project", default="prisma-sae-embeddings")
    parser.add_argument("--wandb-entity", default='akaur64-arizona-state-university')
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seg", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_embeddings = []
    val_embeddings = []
    test_embeddings = []
    for dataset in DATASETS:
        base_path = f"{args.root}/{MODEL}/{dataset}"
        test_embed = load_embeddings(f"{base_path}/valid.pt", False)
        if args.seg:
            if len(test_embed.shape)<=2:
                continue
        else:
            if len(test_embed.shape)>2:
                continue
        train_embeddings.append(load_embeddings(f"{base_path}/train.pt", args.seg))
        val_embeddings.append(load_embeddings(f"{base_path}/valid.pt", args.seg))
        # break
        # test_embeddings.append(load_embeddings(f"{base_path}/test.pt", args.seg))

    train_embeddings = torch.cat(train_embeddings, dim=0)
    val_embeddings = torch.cat(val_embeddings, dim=0)
    # test_embeddings = torch.cat(test_embeddings, dim=0)

    train_loader = EmbeddingDataset(train_embeddings)
    val_loader = EmbeddingDataset(val_embeddings)

    DIM_IN = train_embeddings.shape[-1]
    N_TRAIN = train_embeddings.shape[0]
    N_VAL = val_embeddings.shape[0]
    # delete train_embeddings from memory
    del train_embeddings
    del val_embeddings
    # test_dataset = EmbeddingDataset(test_embeddings)

    train_loader = DataLoader(train_loader, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = build_config(
        d_in=DIM_IN,
        expansion_factor=args.expansion_factor,
        device=str(device),
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoint_path=str(Path(args.output).parent),
    )
    cfg.train_batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.lr = args.lr
    cfg.l1_coefficient = args.l1

    sae = StandardSparseAutoencoder(cfg).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr)

    print(
        "NOTE: INFO lines above are Prisma VisionModelSAERunnerConfig defaults "
        "(e.g. Total training images=1300000 is a hardcoded ImageNet size; "
        "Expansion factor may show 16 before our override). They do not describe this run."
    )
    print("Our hyperparameters:")
    hyperparams = {
        "root": args.root,
        "model": MODEL,
        "datasets": DATASETS,
        "device": str(device),
        "output": args.output,
        "d_in": cfg.d_in,
        "d_sae": cfg.d_sae,
        "expansion_factor": cfg.expansion_factor,
        "architecture": cfg.architecture,
        "activation_fn": cfg.activation_fn_str,
        "normalize_activations": cfg.normalize_activations,
        "b_dec_init_method": cfg.b_dec_init_method,
        "initialization_method": cfg.initialization_method,
        "batch_size": cfg.train_batch_size,
        "epochs": cfg.num_epochs,
        "lr": cfg.lr,
        "l1_coefficient": cfg.l1_coefficient,
        "lp_norm": cfg.lp_norm,
        "seed": cfg.seed,
        "dtype": cfg._dtype,
        "dead_feature_threshold": cfg.dead_feature_threshold,
        "log_to_wandb": cfg.log_to_wandb,
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        # "n_test": test_embeddings.shape[0],
        "steps_per_epoch": len(train_loader),
        "total_train_steps": len(train_loader) * cfg.num_epochs,
    }
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    run = None
    if not args.no_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --no-wandb was not set")
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={
            "d_in": cfg.d_in,
            "d_sae": cfg.d_sae,
            "expansion_factor": cfg.expansion_factor,
            "batch_size": cfg.train_batch_size,
            "epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "l1": cfg.l1_coefficient,
            "device": str(device),
        })

    best_val = math.inf
    for epoch in range(cfg.num_epochs):
        train_stats = train_one_epoch(sae, train_loader, opt, cfg, device)
        val_stats = evaluate(sae, val_loader, cfg, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.6f} val_loss={val_stats['loss']:.6f} "
            f"val_cos={val_stats['cos']:.4f} val_ev_mean={val_stats['ev_mean']:.4f} val_ev_std={val_stats['ev_std']:.4f} val_l0={val_stats['l0']:.1f} val_dead_frac={val_stats['dead_frac']:.4f}"
        )

        if run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/mse": train_stats["mse"],
                "train/l1": train_stats["l1"],
                "train/cosine": train_stats["cos"],
                "train/explained_variance_mean": train_stats["ev_mean"],
                "train/explained_variance_std": train_stats["ev_std"],
                "train/dead_feature_fraction": train_stats["dead_frac"],
                "train/l0": train_stats["l0"],
                "val/loss": val_stats["loss"],
                "val/mse": val_stats["mse"],
                "val/l1": val_stats["l1"],
                "val/cosine": val_stats["cos"],
                "val/explained_variance_mean": val_stats["ev_mean"],
                "val/explained_variance_std": val_stats["ev_std"],
                "val/l0": val_stats["l0"],
                "val/dead_feature_fraction": val_stats["dead_frac"],
            })

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            sae.save_model(args.output)

    if run is not None:
        run.finish()

    print(f"best checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
