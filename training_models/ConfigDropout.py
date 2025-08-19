# ConfigDropout.py
# Configurable random dropout over epochs using decay laws + optional noise
# Plugs into main as: --mode train_with_configurable

import os
import time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset
from torch import optim
from selective_gradient import TrainRevision
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import matplotlib.pyplot as plt  # for debug histogram

# ============================
# HYPERPARAMS (edit & go)
# ============================
# Decays available: negexp, invpower, linear, step, logistic
DECAY_FUNCTION = "invpower"
BETA           = 3.0          # shared aggressiveness/steepness
MIN_KEEP_RATIO = 0.10         # floor on kept fraction (destination)

# Spike pattern: 0=no spikes, 1=final 100%, 2=+mid 50%, 3=+quarter 25%
SPIKE          = 3

# noise (no salt & pepper)
NOISE_TYPE     = "none"   # ["none","gaussian","uniform"]
NOISE_LEVEL    = 0.00         # std/amplitude; if 0 => no noise

VAL_FRAC       = 0.00         # carve from TRAIN once for validation
SEED           = 42           # rng seed for schedules/noise

# ---------- env overrides (used by tuner / CLI runs) ----------
DECAY_FUNCTION = os.getenv("CD_DECAY_FUNCTION", DECAY_FUNCTION)
BETA           = float(os.getenv("CD_BETA", BETA))
MIN_KEEP_RATIO = float(os.getenv("CD_MIN_KEEP_RATIO", MIN_KEEP_RATIO))
SPIKE          = int(os.getenv("CD_SPIKE", SPIKE))
NOISE_TYPE     = os.getenv("CD_NOISE_TYPE", NOISE_TYPE)
NOISE_LEVEL    = float(os.getenv("CD_NOISE_LEVEL", NOISE_LEVEL))
VAL_FRAC       = float(os.getenv("CD_VAL_FRAC", VAL_FRAC))
SEED           = int(os.getenv("CD_SEED", SEED))

# ============================
# helpers
# ============================
class _IndexedSubset(Dataset):
    """Wrap a Subset to also return the original dataset index."""
    def __init__(self, subset: Subset):
        assert isinstance(subset, Subset)
        self.subset = subset
        self.indices = np.array(subset.indices)

    def __len__(self): return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return x, y, int(self.indices[i])


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

    # ---------- tiny split ----------
    def _prepare_val_split(self, train_loader, val_frac, seed):
        ds = train_loader.dataset
        N  = len(ds)
        idx = np.arange(N)
        self.rng.shuffle(idx)
        v = int(N * float(val_frac))
        val_idx, train_idx = idx[:v].tolist(), idx[v:].tolist()

        train_subset = Subset(ds, train_idx)
        # Wrap training so we can read original indices for debug/overlap
        train_indexed = _IndexedSubset(train_subset)

        val_subset   = Subset(ds, val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size if hasattr(self, "batch_size") else getattr(train_loader, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 2),
            pin_memory=True,
        )
        return train_indexed, val_loader

    # ---------- decay laws (use shared BETA, t in [0,1]) ----------
    def _negexp(self, t):    # e^{-β t}
        return math.exp(-BETA * t)

    def _invpower(self, t):  # (1 + t)^{-β}
        return (1.0 + t) ** (-BETA)

    def _linear(self, t):
        """
        Linear decay until cutoff; larger β => faster drop (time scaled).
        f(t) = 1 - min(1, β * t)
        """
        return max(0.0, 1.0 - min(1.0, BETA * t))

    def _step(self, t):
        """
        Simple stepped decay (3 steps). β controls the height of later steps.
        thresholds: 1/3, 2/3.
        """
        if t < 1.0/3.0:
            return 1.0
        elif t < 2.0/3.0:
            return (2.0/3.0) ** max(0.0, BETA)
        else:
            return (1.0/3.0) ** max(0.0, BETA)

    def _logistic(self, t):
        """
        Reversed cumulative logistic: decays from ~1 to ~0.
        β controls steepness (bigger β -> steeper).
        f(t) = 1 / (1 + exp((t - 0.5)/s)),  s = 0.25 / max(1, β)
        """
        s = 0.25 / max(1.0, BETA)
        return 1.0 / (1.0 + math.exp((t - 0.5) / max(1e-8, s)))

    def _base_frac(self, epoch):
        """Return un-noised keep fraction r(t) in [MIN_KEEP_RATIO, 1]."""
        if self.epochs <= 1:
            return 1.0
        # normalize epoch to t ∈ [0,1]
        t = epoch / (self.epochs - 1)

        fsel = DECAY_FUNCTION.lower()
        if fsel in ("negexp", "exp"):
            f = self._negexp(t)
        elif fsel in ("invpower", "power"):
            f = self._invpower(t)
        elif fsel == "linear":
            f = self._linear(t)
        elif fsel == "step":
            f = self._step(t)
        elif fsel == "logistic":
            f = self._logistic(t)
        else:
            raise ValueError(f"unknown DECAY_FUNCTION: {DECAY_FUNCTION}")

        # map f∈[0,1] -> [MIN_KEEP_RATIO, 1]
        f = max(0.0, min(1.0, f))
        return float(MIN_KEEP_RATIO + (1.0 - MIN_KEEP_RATIO) * f)

    # ---------- spikes ----------
    def _spike_override(self, epoch):
        """Return keep fraction override for spike epochs, or None."""
        if self.epochs <= 0 or SPIKE <= 0:
            return None
        last = self.epochs - 1
        mid  = max(0, self.epochs // 2)
        qtr  = max(0, int(round(self.epochs / 4)))
        if epoch == last:
            return 1.0
        if SPIKE >= 2 and epoch == mid:
            return 0.5
        if SPIKE >= 3 and epoch == qtr:
            return 0.25
        return None

    # ---------- noise ----------
    def _apply_noise(self, x, epoch):
        if NOISE_TYPE == "none" or NOISE_LEVEL == 0.0:
            return x
        r = np.random.default_rng(self.seed + 7919 * epoch)
        if NOISE_TYPE == "gaussian":
            return x + r.normal(0.0, NOISE_LEVEL)
        if NOISE_TYPE == "uniform":
            return x + r.uniform(-NOISE_LEVEL, NOISE_LEVEL)
        return x

    def _keep_fraction(self, epoch):
        """
        Return (keep_req, base). Epoch 0 requests full data.
        Spike epochs (if any) override with fixed keep fractions.
        """
        if epoch == 0:
            return 1.0, 1.0

        base = self._base_frac(epoch)
        keep = self._apply_noise(base, epoch)
        keep = float(np.clip(keep, MIN_KEEP_RATIO, 1.0))

        spike = self._spike_override(epoch)
        if spike is not None:
            keep = spike  # exact revision amount

        return keep, base

    # ---------- loader (full dataset; per-batch dropout happens inside loop) ----------
    def _epoch_loader(self):
        """Return a loader over the *full* train set (shuffled)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

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

        print(
            f"[schedule] decay={DECAY_FUNCTION}"
            f" | BETA={BETA}"
            f" | min_keep={MIN_KEEP_RATIO} | spikes={SPIKE}"
            f" | noise={NOISE_TYPE} level={NOISE_LEVEL}"
            f" | val_frac={VAL_FRAC}"
            f" | sampler=per-batch random keep k≈r·B (no mask forward pass)"
        )

        start_time = time.time()
        epoch_train_acc, epoch_test_acc = [], []
        time_per_epoch, samples_used_per_epoch = [], []
        total_samples_bp = 0

        prev_used = None
        overlaps = []  # % overlap(|S_t ∩ S_{t-1}| / |S_t|)

        for epoch in range(self.epochs):
            keep_req, base = self._keep_fraction(epoch)
            force_full = (keep_req >= 0.999)

            loader = self._epoch_loader()
            used_indices = set()  # indices actually trained on this epoch

            # --- train epoch (pure random dropout inside each batch) ---
            self.model.train()
            t0 = time.time()
            running_loss, running_acc, running_n = 0.0, 0.0, 0
            used_count = 0

            iterator = loader
            if tqdm is not None:
                iterator = tqdm(loader, leave=False, dynamic_ncols=True, desc=f"epoch {epoch+1}/{self.epochs}")

            for xb, yb, ib in iterator:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                B = yb.size(0)
                r = 1.0 if force_full else float(np.clip(keep_req, MIN_KEEP_RATIO, 1.0))
                k = max(1, int(round(r * B)))

                # ---- PURE RANDOM DROPOUT (no mask forward pass) ----
                idx = torch.randperm(B, device=xb.device)[:k]
                xb_sel = xb.index_select(0, idx)
                yb_sel = yb.index_select(0, idx)

                # track which dataset items were actually used
                ib_sel = torch.as_tensor(ib, device=idx.device)[idx].tolist()
                used_indices.update(ib_sel)

                out = self.model(xb_sel)
                loss = ce(out, yb_sel)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                acc = (out.argmax(1) == yb_sel).float().mean().item()
                running_loss += loss.item() * k
                running_acc  += acc * k
                running_n    += k
                used_count   += k
                total_samples_bp += k

                if tqdm is not None:
                    iterator.set_postfix(loss=f"{loss.item():.4f}", kept=k)

            epoch_time = time.time() - t0
            train_loss = running_loss / max(1, running_n)
            train_acc  = running_acc  / max(1, running_n)
            epoch_train_acc.append(train_acc)
            samples_used_per_epoch.append(used_count)
            time_per_epoch.append(epoch_time)

            # --- overlap vs previous epoch ---
            if prev_used is None:
                overlap_pct = 0.0
            else:
                inter = len(used_indices & prev_used)
                overlap_pct = (inter / max(1, len(used_indices))) * 100.0
            overlaps.append(overlap_pct)
            prev_used = used_indices

            # --- validation + test ---
            val_loss,  val_acc  = self._eval_once(self.val_loader)
            test_loss, test_acc = self._eval_once(self.test_loader)
            epoch_test_acc.append(test_acc)

            t_norm = 0.0 if self.epochs <= 1 else epoch / (self.epochs - 1)
            print(
                f"[{epoch+1}/{self.epochs}] "
                f"t={t_norm:.3f} base_keep={base:.3f} keep_req={keep_req:.3f} "
                f"used={used_count} | overlap_prev={overlap_pct:5.1f}% | "
                f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"test: loss={test_loss:.4f} acc={test_acc:.4f}"
            )

        end_time = time.time()
        log_memory(start_time, end_time)

        # wall clock summary
        total_sec = end_time - start_time
        mm = int(total_sec // 60); ss = total_sec - 60 * mm
        print(f"Training Time: {mm}m {ss:.2f}s")

        # plots (unchanged signatures)
        plot_accuracy_time_multi(self.model_name + "_config_train", epoch_train_acc, time_per_epoch,
                                 self.save_path, self.save_path)
        plot_accuracy_time_multi_test(self.model_name + "_config_train", epoch_test_acc, time_per_epoch,
                                      samples_used_per_epoch, self.threshold,
                                      self.save_path, self.save_path)

        # --- histogram of overlap percentages ---
        bins = np.linspace(0, 100, 11)
        counts, edges = np.histogram(np.array(overlaps), bins=bins)
        print("\nSampler overlap % between consecutive epochs (|S_t ∩ S_{t-1}| / |S_t|):")
        for i in range(len(counts)):
            lo, hi = int(edges[i]), int(edges[i+1])
            print(f"{lo:2d}-{hi:3d}% : {counts[i]}")
        try:
            os.makedirs(self.save_path, exist_ok=True)
            plt.figure()
            plt.hist(overlaps, bins=10)
            plt.xlabel("Overlap with previous epoch (%)")
            plt.ylabel("Count")
            plt.title("Per-epoch sample overlap (pure per-batch random dropout)")
            out_path = os.path.join(self.save_path, "sampler_overlap_hist.png")
            plt.tight_layout(); plt.savefig(out_path); plt.close()
            print(f"Overlap histogram saved to: {out_path}")
        except Exception as e:
            print(f"(histogram save skipped: {e})")

        # return model + total_samples_backpropped so main's EE calc works
        return self.model, int(total_samples_bp)
