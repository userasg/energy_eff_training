# ConfigDropout.py
# configurable random dropout over epochs using decay laws + optional noise
# plugs into main as: --mode train_with_configurable

import os
import time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch import optim
from selective_gradient import TrainRevision
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ============================
# HYPERPARAMS (edit & go)
# ============================
DECAY_FUNCTION     = "negexp"     # ["negexp", "invpower"]
BETA               = 12.5        # negexp: f(t)=exp(-BETA*t)
POWER_BETA         = 0.0          # invpower: f(t)=(1+t)^(-POWER_BETA)
MIN_KEEP_RATIO     = 0.45         # floor on kept fraction (destination)
FINAL_REVISION     = True         # train on full dataset at last epoch

NOISE_TYPE         = "gaussian"   # ["none","gaussian","uniform","saltpepper"]
NOISE_LEVEL        = 0.02         # std/amplitude; if 0 => no noise
NOISE_PROB         = 0.00         # used by saltpepper

VAL_FRAC           = 0.00         # carve from TRAIN once for validation
SEED               = 42           # rng seed for subset + noise

# NEW: subset formation mode
# - True  => progressive/nested: fix a permutation once; each epoch takes a prefix (carry-over).
# - False => resample: draw a fresh random subset each epoch (original behavior).
PROGRESSIVE_PREFIX = False

def _as_bool(v): return str(v).lower() in ("1", "true", "t", "yes", "y")

# env overrides
DECAY_FUNCTION     = os.getenv("CD_DECAY_FUNCTION", DECAY_FUNCTION)
BETA               = float(os.getenv("CD_BETA", BETA))
POWER_BETA         = float(os.getenv("CD_POWER_BETA", POWER_BETA))
MIN_KEEP_RATIO     = float(os.getenv("CD_MIN_KEEP_RATIO", MIN_KEEP_RATIO))
FINAL_REVISION     = _as_bool(os.getenv("CD_FINAL_REVISION", FINAL_REVISION))
NOISE_TYPE         = os.getenv("CD_NOISE_TYPE", NOISE_TYPE)
NOISE_LEVEL        = float(os.getenv("CD_NOISE_LEVEL", NOISE_LEVEL))
NOISE_PROB         = float(os.getenv("CD_NOISE_PROB", NOISE_PROB))
VAL_FRAC           = float(os.getenv("CD_VAL_FRAC", VAL_FRAC))
SEED               = int(os.getenv("CD_SEED", SEED))
PROGRESSIVE_PREFIX = _as_bool(os.getenv("CD_PROGRESSIVE_PREFIX", PROGRESSIVE_PREFIX))

# ============================
# Trainer
# ============================
class ConfigDropout(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader, device,
                 epochs, save_path, threshold, seed: int = SEED):
        super().__init__(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold)

        self.seed = int(seed)
        self.rng  = np.random.default_rng(self.seed)

        # one-time holdout from provided TRAIN for validation (no main.py changes)
        self.train_dataset, self.val_loader = self._prepare_val_split(train_loader, VAL_FRAC, self.seed)
        self.data_size  = len(self.train_dataset)
        self.batch_size = getattr(train_loader, "batch_size", 32)
        self.num_workers = getattr(train_loader, "num_workers", 2)

        # --- for progressive / nested subsets ---
        self.progressive = bool(PROGRESSIVE_PREFIX)
        if self.progressive:
            self.perm = self.rng.permutation(self.data_size)  # fixed permutation
            self._prev_n_keep = self.data_size                # track last prefix length

    # ---------- tiny split ----------
    def _prepare_val_split(self, train_loader, val_frac, seed):
        ds = train_loader.dataset
        N  = len(ds)
        idx = np.arange(N)
        self.rng.shuffle(idx)
        v = int(N * float(val_frac))
        val_idx, train_idx = idx[:v].tolist(), idx[v:].tolist()

        train_subset = Subset(ds, train_idx)
        val_subset   = Subset(ds, val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size if hasattr(self, "batch_size") else getattr(train_loader, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 2),
            pin_memory=True,
        )
        return train_subset, val_loader

    # ---------- decay laws ----------
    def _negexp(self, t):         # t∈[0,1]
        return math.exp(-BETA * t)
    def _invpower(self, t):       # f(0)=1, monotone ↓
        return (1.0 + t) ** (-POWER_BETA)

    def _base_frac(self, epoch):
        if self.epochs <= 1: return 1.0
        t = epoch / (self.epochs - 1)
        if DECAY_FUNCTION.lower() in ["negexp", "exp"]:
            f = self._negexp(t)
        elif DECAY_FUNCTION.lower() in ["invpower", "power"]:
            f = self._invpower(t)
        else:
            raise ValueError(f"unknown DECAY_FUNCTION: {DECAY_FUNCTION}")
        # map f∈(0,1] -> [MIN_KEEP_RATIO, 1]
        return float(MIN_KEEP_RATIO + (1.0 - MIN_KEEP_RATIO) * f)

    # ---------- noise ----------
    def _apply_noise(self, x, epoch):
        if NOISE_TYPE == "none" or NOISE_LEVEL == 0.0: return x
        r = np.random.default_rng(self.seed + 7919 * epoch)
        if NOISE_TYPE == "gaussian":
            return x + r.normal(0.0, NOISE_LEVEL)
        if NOISE_TYPE == "uniform":
            return x + r.uniform(-NOISE_LEVEL, NOISE_LEVEL)
        if NOISE_TYPE in ["saltpepper", "salt_pepper", "salt-and-pepper"]:
            return 1.0 if r.uniform() < NOISE_PROB else MIN_KEEP_RATIO
        return x

    def _keep_fraction(self, epoch):
        """Return (keep_req, base). Epoch 0 and final epoch (if enabled) request full data."""
        base = self._base_frac(epoch)
        if epoch == 0:
            return 1.0, base
        if FINAL_REVISION and epoch == self.epochs - 1:
            return 1.0, base
        keep = self._apply_noise(base, epoch)
        keep = float(np.clip(keep, MIN_KEEP_RATIO, 1.0))
        return keep, base

    # ---------- per-epoch subset loader ----------
    def _epoch_loader(self, frac, *, force_full=False):
        """Build DataLoader for current epoch.
        force_full=True overrides the progressive prefix rule and re-expands to all data.
        """
        if force_full:
            n_keep = self.data_size
            if self.progressive:
                idx = self.perm.tolist()  # use entire fixed order
                self._prev_n_keep = self.data_size
            else:
                idx = self.rng.permutation(self.data_size).tolist()
        else:
            n_target = max(1, int(round(frac * self.data_size)))
            if self.progressive:
                n_keep = min(n_target, self._prev_n_keep)
                idx = self.perm[:n_keep].tolist()
                self._prev_n_keep = n_keep
            else:
                n_keep = n_target
                idx = self.rng.permutation(self.data_size)[:n_keep].tolist()

        subset = Subset(self.train_dataset, idx)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        ), n_keep

    # ---------- eval ----------
    def _eval_once(self, loader):
        self.model.eval()
        loss_tot, acc_tot, n_tot = 0.0, 0.0, 0
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                out = self.model(x)
                loss = ce(out, y)
                pred = out.argmax(1)
                acc = (pred == y).float().mean().item()
                b = y.size(0)
                n_tot += b
                loss_tot += loss.item() * b
                acc_tot  += acc * b
        return loss_tot / max(1, n_tot), acc_tot / max(1, n_tot)

    # ---------- main ----------
    def train_with_configurable(self):
        self.model.to(self.device)
        ce  = nn.CrossEntropyLoss()
        opt = optim.AdamW(self.model.parameters(), lr=3e-4)

        subset_mode = "progressive-prefix" if self.progressive else "resample-each-epoch"
        print(
            f"[schedule] decay={DECAY_FUNCTION}"
            f" | BETA={BETA} POWER_BETA={POWER_BETA}"
            f" | min_keep={MIN_KEEP_RATIO} final_revision={FINAL_REVISION}"
            f" | noise={NOISE_TYPE} level={NOISE_LEVEL} p={NOISE_PROB}"
            f" | val_frac={VAL_FRAC}"
            f" | subset_mode={subset_mode}"
        )

        start_time = time.time()
        epoch_train_acc, epoch_test_acc = [], []
        time_per_epoch, samples_used_per_epoch = [], []
        total_samples_bp = 0

        for epoch in range(self.epochs):
            keep_req, base = self._keep_fraction(epoch)
            force_full = FINAL_REVISION and (epoch == self.epochs - 1)
            loader, n_keep = self._epoch_loader(keep_req, force_full=force_full)
            samples_used_per_epoch.append(n_keep)

            # --- train epoch ---
            self.model.train()
            t0 = time.time()
            running_loss, running_acc, running_n = 0.0, 0.0, 0

            iterator = loader
            if tqdm is not None:
                iterator = tqdm(loader, leave=False, dynamic_ncols=True, desc=f"epoch {epoch+1}/{self.epochs}")

            for xb, yb in iterator:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                out = self.model(xb)
                loss = ce(out, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bsz = yb.size(0)
                acc = (out.argmax(1) == yb).float().mean().item()
                running_loss += loss.item() * bsz
                running_acc  += acc * bsz
                running_n    += bsz
                total_samples_bp += bsz

                if tqdm is not None:
                    iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

            epoch_time = time.time() - t0
            train_loss = running_loss / max(1, running_n)
            train_acc  = running_acc  / max(1, running_n)
            epoch_train_acc.append(train_acc)

            # --- validation + test ---
            val_loss,  val_acc  = self._eval_once(self.val_loader)
            test_loss, test_acc = self._eval_once(self.test_loader)
            epoch_test_acc.append(test_acc)
            time_per_epoch.append(epoch_time)

            t_norm = 0.0 if self.epochs <= 1 else epoch / (self.epochs - 1)
            print(
                f"[{epoch+1}/{self.epochs}] "
                f"t={t_norm:.3f} base_keep={base:.3f} keep_req={keep_req:.3f} n={n_keep} | "
                f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"test: loss={test_loss:.4f} acc={test_acc:.4f}"
            )

        end_time = time.time()
        log_memory(start_time, end_time)

        # wall clock summary
        total_sec = end_time - start_time
        mm = int(total_sec // 60)
        ss = total_sec - 60 * mm
        print(f"Training Time: {mm}m {ss:.2f}s")

        # plots (unchanged signatures)
        plot_accuracy_time_multi(self.model_name + "_config_train", epoch_train_acc, time_per_epoch,
                                 self.save_path, self.save_path)
        plot_accuracy_time_multi_test(self.model_name + "_config_train", epoch_test_acc, time_per_epoch,
                                      samples_used_per_epoch, self.threshold,
                                      self.save_path, self.save_path)

        # return model + total_samples_backpropped so main's EE calc works
        return self.model, int(total_samples_bp)
