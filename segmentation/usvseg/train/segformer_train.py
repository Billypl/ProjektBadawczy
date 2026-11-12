"""SegFormer training on LaRS (HF transformers, own loop).

    python -m usvseg.train.segformer_train --config configs/m1_segformer_b2.yaml
    python -m usvseg.train.segformer_train --config configs/m1_segformer_b2.yaml --smoke
    python -m usvseg.train.segformer_train --config configs/m1_segformer_b2.yaml --resume

Outputs in runs/<exp>/: config.yaml, metrics.csv, checkpoint/ (last epoch, HF format),
checkpoint_best/ (best dev mIoU so far), state.pt (model + optimizer + scheduler + RNG
after the last finished epoch). Stop the run at any time (kill / Ctrl-C) and restart
with --resume: it continues from the last finished epoch, so at most one epoch is lost.
Dev mIoU is logged for monitoring only; reported numbers come from lars_evaluator on val.
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from common.interfaces import IGNORE
from usvseg.data.lars import Lars, LarsSample
from usvseg.models.segformer import build_segformer
from usvseg.train.transforms import TrainTransform, keep_ratio_scale, normalize, resize_pair

REPO = Path(__file__).resolve().parents[2]
NUM_CLASSES = 3


class SegDataset(Dataset):
    def __init__(self, samples: list[LarsSample], transform: TrainTransform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        # worker seed comes from the DataLoader generator, so runs are reproducible
        rng = np.random.default_rng([torch.initial_seed() % 2**32, i])
        s = self.samples[i]
        return self.transform(s.image(), s.semantic(), rng)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def param_groups(model, lr: float, weight_decay: float, head_lr_mult: float):
    """mmseg paramwise_cfg: no decay for norm/pos_block, 10x lr for the decode head."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = name.startswith("decode_head")
        no_decay = p.ndim == 1 or "norm" in name or "pos_block" in name
        key = (head, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * (head_lr_mult if head else 1.0),
                           "weight_decay": 0.0 if no_decay else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


def poly_with_warmup(total: int, warmup: int, power: float = 1.0):
    def f(it: int) -> float:
        if it < warmup:
            return 1e-6 + (1 - 1e-6) * it / warmup
        return (1 - (it - warmup) / max(1, total - warmup)) ** power
    return f


@torch.no_grad()
def dev_miou(model, samples: list[LarsSample], scale, device) -> tuple[float, list[float]]:
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for s in samples:
        img, gt = s.image(), torch.from_numpy(s.semantic()).to(device).long()
        small, _ = resize_pair(img, None, keep_ratio_scale(img.shape[:2], scale))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixel_values=normalize(small)[None].to(device)).logits
        pred = F.interpolate(logits.float(), size=gt.shape, mode="bilinear", align_corners=False).argmax(1)[0]
        valid = gt != IGNORE
        conf += torch.bincount(gt[valid] * NUM_CLASSES + pred[valid], minlength=NUM_CLASSES**2).view(NUM_CLASSES, -1)
    model.train()
    iou = (conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)).tolist()
    return 100 * float(np.mean(iou)), [100 * v for v in iou]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="20 images, 1 epoch, separate exp dir")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<exp>/state.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    exp = cfg["exp"] + ("_smoke" if args.smoke else "")
    if args.smoke:
        cfg.update(epochs=1, limit=20, dev_limit=5, eval_every=1)
    out = REPO / "runs" / exp
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if yaml.safe_load((out / "config.yaml").read_text()) != cfg:
            raise SystemExit(f"config differs from {out / 'config.yaml'}, refusing to resume")
    elif (out / "state.pt").exists():
        raise SystemExit(f"{out / 'state.pt'} exists: use --resume or remove the run dir")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    seed_everything(cfg["seed"])
    device = torch.device("cuda")

    train = Lars("train").samples
    dev = Lars("dev").samples
    if cfg.get("limit"):
        train = train[: cfg["limit"]]
    if cfg.get("dev_limit"):
        dev = dev[: cfg["dev_limit"]]

    aug = cfg["augment"]
    transform = TrainTransform(base_scale=aug["base_scale"], ratio_range=aug["ratio_range"],
                               crop_hw=aug["crop_hw"], cat_max_ratio=aug["cat_max_ratio"],
                               rotate_deg=aug["rotate_deg"], color_jitter=aug["color_jitter"])
    # not persistent: workers are re-created each epoch, so their seeds come from the
    # generator state, which is saved in state.pt and makes a resumed run identical
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(SegDataset(train, transform), batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], drop_last=True, pin_memory=True, generator=gen)

    model = build_segformer(cfg["encoder"]).to(device).train()
    opt = cfg["optim"]
    optimizer = torch.optim.AdamW(param_groups(model, opt["lr"], opt["weight_decay"], opt["head_lr_mult"]),
                                  lr=opt["lr"], betas=(0.9, 0.999))
    total = cfg["epochs"] * len(loader)
    warmup = min(opt["warmup_iters"], total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_with_warmup(total, warmup))

    print(f"exp={exp} train={len(train)} dev={len(dev)} iters/epoch={len(loader)} total={total} warmup={warmup}",
          flush=True)
    it, start_epoch, best_miou = 0, 1, -1.0
    if args.resume:
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        gen.set_state(st["loader_gen"])
        torch.set_rng_state(st["torch_rng"])
        it, start_epoch, best_miou = st["iter"], st["epoch"] + 1, st["best_miou"]
        print(f"resumed after epoch {st['epoch']} (iter {it}, best dev mIoU {best_miou})", flush=True)

    metrics = open(out / "metrics.csv", "a" if args.resume else "w", newline="")
    log = csv.writer(metrics, lineterminator="\n")
    if not args.resume:
        log.writerow(["epoch", "iter", "loss", "lr", "dev_mIoU", "dev_IoU_obstacle", "dev_IoU_water", "dev_IoU_sky",
                      "epoch_time_s"])

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=x, labels=y).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at iter {it}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sched.step()
            losses.append(loss.item())
            it += 1
            if it % 50 == 0:
                print(f"epoch {epoch} iter {it}/{total} loss {np.mean(losses[-50:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        row = [epoch, it, round(float(np.mean(losses)), 4), f"{sched.get_last_lr()[0]:.3e}", "", "", "", ""]
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            miou, ious = dev_miou(model, dev, cfg["infer_scale"], device)
            row[4:8] = [round(miou, 2)] + [round(v, 2) for v in ious]
            if miou > best_miou:
                best_miou = miou
                model.save_pretrained(out / "checkpoint_best")
        row.append(round(time.time() - t0, 1))
        log.writerow(row)
        metrics.flush()
        print(f"== epoch {epoch} loss {row[2]} dev_mIoU {row[4]} ({row[-1]} s)", flush=True)
        model.save_pretrained(out / "checkpoint")
        tmp = out / "state.pt.tmp"  # write then rename, so a kill during the save cannot corrupt it
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict(), "loader_gen": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch, "iter": it,
                    "best_miou": best_miou}, tmp)
        tmp.replace(out / "state.pt")

    metrics.close()
    print(f"done, checkpoint -> {out / 'checkpoint'}", flush=True)


if __name__ == "__main__":
    main()
