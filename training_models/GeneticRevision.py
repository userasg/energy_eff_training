# GeneticRevision.py
# GA over batch-local dropout schedules (few knobs; self-adaptive ES mutation).
# Now with per-epoch batch report + homogeneity %. Plug-compatible with your main.py.

import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ------------------------------------------------------------------
# Minimal knobs / global defaults (main.py can override KEEP_FLOOR)
# ------------------------------------------------------------------
LR = 3e-4
FINAL_REVISION = False            # if True, last epoch trains with keep=1.00 for every batch
KEEP_FLOOR = 0.35                 # main.py may set GA.KEEP_FLOOR = args.threshold

# Engineering grid / clamps
PARAM_STEP = 0.10
BETA_CLAMP_NEGEXP   = (0.10, 6.00)
BETA_CLAMP_INVPOWER = (0.10, 4.00)
NOISE_LVL_CLAMP     = (0.00, 0.50)

# Init ranges (one-time sampling, not tuned per run)
BETA_INIT_NEGEXP    = (0.50, 3.00)
BETA_INIT_INVPOWER  = (0.50, 2.00)
FLOOR_INIT_RANGE    = (0.30, 0.70)    # actual lower bound uses max(KEEP_FLOOR, FLOOR_INIT_RANGE[0])
NOISE_LVL_INIT      = (0.00, 0.30)

# ES self-adaptation constants (n strategy params = 3: sigma_beta, sigma_floor, sigma_noise)
N_STRAT    = 3
TAU_GLOBAL = 1.0 / (2.0 * N_STRAT) ** 0.5           # τ' = 1/sqrt(2n)
TAU_LOCAL  = 1.0 / (2.0 * (N_STRAT ** 0.5)) ** 0.5  # τ  = 1/sqrt(2*sqrt(n))

# ---------------- utilities ----------------
def _snap(x: float, lo: float, hi: float, step: float = PARAM_STEP) -> float:
    x = float(max(lo, min(hi, x)))
    q = round(x / step) * step
    return float(round(max(lo, min(hi, q)), 2))

def _law_value(law: str, beta: float, t: float) -> float:
    # t in [0,1]
    t = float(max(0.0, min(1.0, t)))
    if law == "negexp":
        # f(t) = exp(-beta * t)
        return math.exp(-beta * t)
    # law == "invpower": f(t) = (1+t)^(-beta)
    return (1.0 + t) ** (-beta)

@dataclass
class Chromosome:
    # phenotypic genes
    law: str                     # "negexp" or "invpower"
    beta: float                  # >= 0.1 (snapped & clamped)
    noise_type: str              # "none" | "gaussian" | "uniform"
    noise_level: float           # [0, 0.5]
    keep_floor: float            # [KEEP_FLOOR, 0.95]

    # strategy (self-adaptive step sizes)
    sigma_beta: float
    sigma_floor: float
    sigma_noise: float

    # per-epoch schedule → keep fraction (2dp)
    def keep_fraction(self, epoch_idx: int, total_epochs: int, rng: random.Random) -> float:
        if total_epochs <= 1:
            base = 1.0
        else:
            t = epoch_idx / float(max(1, total_epochs - 1))
            base = _law_value(self.law, self.beta, t)
        # map base∈(0,1] → [floor, 1]
        keep = self.keep_floor + (1.0 - self.keep_floor) * base

        # optional noise
        if self.noise_type == "gaussian" and self.noise_level > 0.0:
            keep += rng.gauss(0.0, self.noise_level)
        elif self.noise_type == "uniform" and self.noise_level > 0.0:
            keep += rng.uniform(-self.noise_level, self.noise_level)

        keep = max(self.keep_floor, min(1.0, keep))
        return float(round(keep, 2))

# ------------- GA primitives -------------
def _init_chrom(rng: random.Random) -> Chromosome:
    law = "negexp" if rng.random() < 0.5 else "invpower"
    if law == "negexp":
        beta = _snap(rng.uniform(*BETA_INIT_NEGEXP), *BETA_CLAMP_NEGEXP)
    else:
        beta = _snap(rng.uniform(*BETA_INIT_INVPOWER), *BETA_CLAMP_INVPOWER)

    noise_type = ["none", "gaussian", "uniform"][rng.randrange(3)]
    noise_level = _snap(rng.uniform(*NOISE_LVL_INIT), *NOISE_LVL_CLAMP)

    # floor lower bound honors global KEEP_FLOOR
    floor_lo = max(KEEP_FLOOR, FLOOR_INIT_RANGE[0])
    keep_floor = _snap(rng.uniform(floor_lo, FLOOR_INIT_RANGE[1]), KEEP_FLOOR, 0.95)

    # start with modest step sizes (snapped)
    sig_beta  = _snap(0.10, 0.01, 1.00)
    sig_floor = _snap(0.05, 0.01, 0.50)
    sig_noise = _snap(0.05, 0.00, 0.50)
    return Chromosome(law, beta, noise_type, noise_level, keep_floor, sig_beta, sig_floor, sig_noise)

def _recombine(a: Chromosome, b: Chromosome, rng: random.Random) -> Chromosome:
    # discrete genes: coin-flip
    law = a.law if rng.random() < 0.5 else b.law
    noise_type = a.noise_type if rng.random() < 0.5 else b.noise_type
    # continuous: arithmetic mean; step sizes: geometric mean
    beta  = (a.beta + b.beta) / 2.0
    floor = (a.keep_floor + b.keep_floor) / 2.0
    nlvl  = (a.noise_level + b.noise_level) / 2.0
    sb = (a.sigma_beta  * b.sigma_beta ) ** 0.5
    sf = (a.sigma_floor * b.sigma_floor) ** 0.5
    sn = (a.sigma_noise * b.sigma_noise) ** 0.5

    # clamp/snap
    if law == "negexp":
        beta = _snap(beta, *BETA_CLAMP_NEGEXP)
    else:
        beta = _snap(beta, *BETA_CLAMP_INVPOWER)
    floor = _snap(floor, KEEP_FLOOR, 0.95)
    nlvl  = _snap(nlvl,  *NOISE_LVL_CLAMP)
    sb = _snap(sb, 0.01, 1.00)
    sf = _snap(sf, 0.01, 0.50)
    sn = _snap(sn, 0.00, 0.50)

    return Chromosome(law, beta, noise_type, nlvl, floor, sb, sf, sn)

def _mutate_es(c: Chromosome, epoch: int, total_epochs: int, rng: random.Random) -> Chromosome:
    # 1) self-adapt step sizes (log-normal; ES standard)
    g = rng.gauss(0.0, 1.0)  # shared global term
    sb = c.sigma_beta  * math.exp(TAU_GLOBAL * g + TAU_LOCAL * rng.gauss(0.0,1.0))
    sf = c.sigma_floor * math.exp(TAU_GLOBAL * g + TAU_LOCAL * rng.gauss(0.0,1.0))
    sn = c.sigma_noise * math.exp(TAU_GLOBAL * g + TAU_LOCAL * rng.gauss(0.0,1.0))
    sb = _snap(sb, 0.01, 1.00)
    sf = _snap(sf, 0.01, 0.50)
    sn = _snap(sn, 0.00, 0.50)

    # 2) mutate object-level parameters using (possibly) new step sizes
    beta  = c.beta        + sb * rng.gauss(0.0, 1.0)
    floor = c.keep_floor  + sf * rng.gauss(0.0, 1.0)
    nlvl  = c.noise_level + sn * rng.gauss(0.0, 1.0)

    if c.law == "negexp":
        beta = _snap(beta, *BETA_CLAMP_NEGEXP)
    else:
        beta = _snap(beta, *BETA_CLAMP_INVPOWER)
    floor = _snap(floor, KEEP_FLOOR, 0.95)
    nlvl  = _snap(nlvl,  *NOISE_LVL_CLAMP)

    # 3) annealed flips for discrete genes (simple heuristic)
    p_flip = 1.0 / (1.0 + float(epoch))
    law = c.law if rng.random() > p_flip else ("invpower" if c.law == "negexp" else "negexp")
    noise_type = c.noise_type if rng.random() > p_flip else ["none", "gaussian", "uniform"][rng.randrange(3)]

    return Chromosome(law, beta, noise_type, nlvl, floor, sb, sf, sn)

# ---------------- main trainer ----------------
class GeneticRevision:
    """
    Batch-local GA dropout:
      - One chromosome per batch.
      - For epoch e, that batch keeps 'keep_fraction(e)' of its items (randomly) → trains on the kept subset only.
      - Fitness for a chromosome = that batch's mean CE loss.
      - Replacement: keep best half of parents; replace the rest with children (elitism + steady state).
    Works with main.py (expects val_loader and test_loader).
    Returns (model, total_samples_backpropped).
    """
    def __init__(self, model_name, model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device, epochs: int,
                 save_path=None, seed: int = 42):
        self.model_name  = model_name
        self.model       = model
        self.train_loader= train_loader
        self.val_loader  = val_loader
        self.test_loader = test_loader
        self.device      = device
        self.epochs      = int(epochs)
        self.save_path   = save_path

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # One chromosome per *training batch*
        self.num_batches = len(train_loader)
        self.pop: List[Chromosome] = [_init_chrom(self.rng) for _ in range(self.num_batches)]

    @torch.no_grad()
    def _eval_loss_acc(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        crit = nn.CrossEntropyLoss(reduction="sum")
        ce_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out = self.model(xb)
            ce_sum += crit(out, yb).item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.numel()
        self.model.train()
        loss = 0.0 if total == 0 else ce_sum / total
        acc  = 0.0 if total == 0 else correct / total
        return round(loss, 4), round(acc, 4)

    @staticmethod
    def _drop_batch(xb: torch.Tensor, yb: torch.Tensor, keep_frac: float):
        n = yb.size(0)
        k = max(1, int(round(keep_frac * n)))
        if k >= n:  # nothing to drop
            return xb, yb
        idx = torch.randperm(n, device=yb.device)[:k]
        return xb.index_select(0, idx), yb.index_select(0, idx)

    # --------- reporting helpers (added) ---------
    def _homogeneity_pct(self, rows: List[Dict]) -> float:
        # bucket by (law, beta, floor, noise_type, noise_level) rounded to 2dp
        buckets: Dict[Tuple, int] = {}
        for r in rows:
            key = (
                r["law"],
                round(r["beta"], 2),
                round(r["floor"], 2),
                r["noise_type"],
                round(r["noise_level"], 2),
            )
            buckets[key] = buckets.get(key, 0) + 1
        if not buckets:
            return 0.0
        maj = max(buckets.values())
        return round(100.0 * maj / max(1, len(rows)), 2)

    def _print_epoch_report(self, epoch: int, rows: List[Dict], val_loss: float, val_acc: float):
        homo = self._homogeneity_pct(rows)
        kept_total = sum(r["kept"] for r in rows)
        print(f"\n=== Epoch {epoch} Report ===  val: loss={val_loss:.4f} acc={val_acc:.4f}")
        print(f"Samples Kept (for backprop): {kept_total} | Homogeneity: {homo:.2f}%")

        header = (
            " bidx | law      | beta  | floor | noise(type, lvl) | keep  | n_batch | kept | dropped | fitness "
        )
        print(header); print("-" * len(header))
        for r in rows:
            print(
                f" {r['bidx']:4d} | {r['law']:<8s} | {r['beta']:5.2f} | {r['floor']:5.2f} | "
                f"{r['noise_type']:<7s},{r['noise_level']:>4.2f} | {r['keep']:5.2f} | "
                f"{r['n']:7d} | {r['kept']:4d} | {r['drop']:7d} | {r['fitness']:7.4f}"
            )
        print("=" * len(header))

    def train_with_genetic(self):
        dev = self.device
        self.model.to(dev)
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        ce = nn.CrossEntropyLoss(reduction="mean")

        total_samples_bp = 0

        for epoch in range(1, self.epochs + 1):
            # -------- Train over batches; collect fitness per batch --------
            fitness = [None] * self.num_batches  # CE loss per batch (lower is better)
            epoch_rows: List[Dict] = []          # rows for reporting

            iterator = enumerate(self.train_loader)
            if tqdm is not None:
                iterator = tqdm(enumerate(self.train_loader), total=self.num_batches,
                                desc=f"[Batch-GA] Epoch {epoch}/{self.epochs}", leave=False, dynamic_ncols=True)

            self.model.train()
            for bidx, (xb, yb) in iterator:
                chrom = self.pop[bidx]
                keep = 1.0 if (FINAL_REVISION and epoch == self.epochs) \
                        else chrom.keep_fraction(epoch - 1, self.epochs, self.rng)

                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)

                n = int(yb.size(0))
                xb_k, yb_k = self._drop_batch(xb, yb, keep)
                k = int(yb_k.size(0))

                out = self.model(xb_k)
                loss = ce(out, yb_k)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                fit = float(round(loss.item(), 4))
                fitness[bidx] = fit
                total_samples_bp += k

                # collect report row
                epoch_rows.append({
                    "bidx": bidx,
                    "law": chrom.law,
                    "beta": float(round(chrom.beta, 2)),
                    "floor": float(round(chrom.keep_floor, 2)),
                    "noise_type": chrom.noise_type,
                    "noise_level": float(round(chrom.noise_level, 2)),
                    "keep": float(round(keep, 2)),
                    "n": n,
                    "kept": k,
                    "drop": n - k,
                    "fitness": fit,
                })

                if tqdm is not None:
                    iterator.set_postfix(keep=f"{keep:.2f}", loss=f"{loss.item():.4f}")

            # -------- Validation (full-batch, no dropout) --------
            val_loss, val_acc = self._eval_loss_acc(self.val_loader)

            # -------- (μ+λ) selection with elitism (no extra passes) --------
            children: List[Chromosome] = []
            for _ in range(self.num_batches):
                i = self.rng.randrange(self.num_batches)
                j = self.rng.randrange(self.num_batches)
                child = _recombine(self.pop[i], self.pop[j], self.rng)
                child = _mutate_es(child, epoch, self.epochs, self.rng)
                children.append(child)

            order = sorted(range(self.num_batches), key=lambda i: fitness[i])
            survivors_idx = order[: self.num_batches // 2]                # best half (parents)
            survivors = [self.pop[i] for i in survivors_idx]
            inject = children[: self.num_batches - len(survivors)]        # fill with children
            self.pop = survivors + inject

            # -------- Epoch report (NEW) --------
            self._print_epoch_report(epoch, epoch_rows, val_loss, val_acc)

        # -------- Final test --------
        test_loss, test_acc = self._eval_loss_acc(self.test_loader)
        print(f"\n=== FINAL TEST ===  loss={test_loss:.4f} acc={test_acc:.4f}")

        return self.model, total_samples_bp
