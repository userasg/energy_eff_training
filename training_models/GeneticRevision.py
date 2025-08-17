# GeneticRevision.py
# Simple GA over a single decay schedule (inverse power).
# Evolved genes: beta (schedule steepness) and gaussian noise level.
# Compatible with main.py. Returns (model, total_samples_backpropped).

import math, random, time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import plot_accuracy_time_multi, plot_accuracy_time_multi_test

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------------- minimal knobs ----------------
LR = 3e-4
KEEP_FLOOR = 0.35         # lower bound on keep fraction
FINAL_REVISION = False    # if True, last epoch keeps 1.00 for all batches
VPOP = 8                  # virtual population size (small)
HARD_CONVERGE_LAST_K = 0  # clone best across population in the final K epochs (0 to disable)

# numeric domains
PARAM_STEP = 0.5
BETA_CLAMP_INVPOWER = (1.00, 15.00)
NOISE_LVL_CLAMP     = (0.00, 0.50)

# init ranges
BETA_INIT_INVPOWER  = (1.00, 15.00)
NOISE_LVL_INIT      = (0.00, 0.50)

# ES self-adaptation (n=2 step-sizes)
N_STRAT    = 2
TAU_GLOBAL = 1.0 / (2.0 ** 0.5)
TAU_LOCAL  = 1.0 / (2.0 * (N_STRAT ** 0.5))

def _snap(x: float, lo: float, hi: float, step: float = PARAM_STEP) -> float:
    x = float(max(lo, min(hi, x)))
    q = round(x / step) * step
    return float(round(max(lo, min(hi, q)), 2))

def _invpower(beta: float, t: float) -> float:
    # t in [0,1]; f(t) = (1+t)^(-beta)
    t = float(max(0.0, min(1.0, t)))
    return (1.0 + t) ** (-beta)

def _anneal(p: float) -> float:
    # 1 -> 0; squared for stronger squeeze late
    return max(0.0, 1.0 - p) ** 2

def _survivor_frac(p: float) -> float:
    # increase selection pressure 50% -> 25%
    return 0.25 + 0.25 * (1.0 - p)

def _reshuffle_prob(p: float) -> float:
    return max(0.0, 1.0 - p)

def _noise_scale(p: float) -> float:
    return _anneal(p)

@dataclass
class Chromosome:
    # fixed law: invpower
    beta: float               # schedule steepness
    noise_level: float        # gaussian noise std on keep fraction
    sigma_beta: float         # ES step size for beta
    sigma_noise: float        # ES step size for noise_level

    def keep_fraction(self, epoch_idx: int, total_epochs: int, rng: random.Random) -> float:
        # base schedule
        p = 1.0 if total_epochs <= 1 else epoch_idx / float(max(1, total_epochs - 1))
        base = _invpower(self.beta, p)            # in (0,1]
        keep = KEEP_FLOOR + (1.0 - KEEP_FLOOR) * base

        # annealed gaussian noise
        eff_noise = self.noise_level * _noise_scale(p)
        if eff_noise > 0.0:
            keep += rng.gauss(0.0, eff_noise)

        keep = max(KEEP_FLOOR, min(1.0, keep))
        return float(round(keep, 2))

def _init_chrom(rng: random.Random) -> Chromosome:
    beta = _snap(rng.uniform(*BETA_INIT_INVPOWER), *BETA_CLAMP_INVPOWER)
    nlvl = _snap(rng.uniform(*NOISE_LVL_INIT), *NOISE_LVL_CLAMP)
    sb   = _snap(0.10, 0.01, 1.00)
    sn   = _snap(0.05, 0.00, 0.50)
    return Chromosome(beta, nlvl, sb, sn)

def _recombine(a: Chromosome, b: Chromosome, rng: random.Random) -> Chromosome:
    # arithmetic mean on genes; geometric mean on step sizes
    beta = (a.beta + b.beta) / 2.0
    nlvl = (a.noise_level + b.noise_level) / 2.0
    sb   = (a.sigma_beta  * b.sigma_beta ) ** 0.5
    sn   = (a.sigma_noise * b.sigma_noise) ** 0.5
    beta = _snap(beta, *BETA_CLAMP_INVPOWER)
    nlvl = _snap(nlvl, *NOISE_LVL_CLAMP)
    sb   = _snap(sb, 0.01, 1.00)
    sn   = _snap(sn, 0.00, 0.50)
    return Chromosome(beta, nlvl, sb, sn)

def _mutate_es(c: Chromosome, epoch: int, total_epochs: int, rng: random.Random) -> Chromosome:
    p = epoch / float(max(1, total_epochs))
    a = _anneal(p)

    # self-adapt step sizes (log-normal), then anneal
    g  = rng.gauss(0.0, 1.0)
    sb = c.sigma_beta  * math.exp(TAU_GLOBAL * g + TAU_LOCAL * rng.gauss(0.0,1.0)) * a
    sn = c.sigma_noise * math.exp(TAU_GLOBAL * g + TAU_LOCAL * rng.gauss(0.0,1.0)) * a
    sb = _snap(sb, 0.00, 1.00)
    sn = _snap(sn, 0.00, 0.50)

    # mutate object parameters
    beta = c.beta        + (sb * rng.gauss(0.0, 1.0) if sb > 0 else 0.0)
    nlvl = c.noise_level + (sn * rng.gauss(0.0, 1.0) if sn > 0 else 0.0)
    beta = _snap(beta, *BETA_CLAMP_INVPOWER)
    nlvl = _snap(nlvl, *NOISE_LVL_CLAMP)

    return Chromosome(beta, nlvl, sb, sn)

class GeneticRevision:
    """
    V-pop GA over batch-local keep fractions using a single invpower law.
    Genes: beta and gaussian noise_level.
    """

    def __init__(self, model_name, model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device, epochs: int,
                 save_path=None, seed: int = 42):
        self.model_name = model_name
        self.model      = model
        self.train_loader= train_loader
        self.val_loader  = val_loader
        self.test_loader = test_loader
        self.device     = device
        self.epochs     = int(epochs)
        self.save_path  = save_path

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.num_batches = len(train_loader)
        self.vpop = min(VPOP, max(1, self.num_batches))
        self.pop: List[Chromosome] = [_init_chrom(self.rng) for _ in range(self.vpop)]
        self.assign = [i % self.vpop for i in range(self.num_batches)]

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
        if k >= n:
            return xb, yb, 0
        idx = torch.randperm(n, device=yb.device)[:k]
        dropped = int(n - k)
        return xb.index_select(0, idx), yb.index_select(0, idx), dropped

    def train_with_genetic(self):
        dev = self.device
        self.model.to(dev)
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        ce  = nn.CrossEntropyLoss(reduction="mean")

        total_samples_bp = 0
        epoch_train_acc: List[float] = []
        val_acc_hist: List[float] = []
        time_per_epoch: List[float] = []
        samples_used_per_epoch: List[int] = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            p  = (epoch - 1) / float(max(1, self.epochs - 1))
            reshuffle_p = _reshuffle_prob(p)
            surv_frac   = _survivor_frac(p)

            # aggregate per-chromosome stats
            per_chrom = [{"loss_sum":0.0,"batches":0,"samps_kept":0,"dropped":0} for _ in range(self.vpop)]
            run_loss_sum, run_correct, run_total = 0.0, 0, 0

            iterator = enumerate(self.train_loader)
            if tqdm is not None:
                iterator = tqdm(enumerate(self.train_loader), total=self.num_batches,
                                desc=f"[GA(invpower)] Epoch {epoch}/{self.epochs}", leave=False, dynamic_ncols=True)

            self.model.train()
            for bidx, (xb, yb) in iterator:
                cid   = self.assign[bidx]
                chrom = self.pop[cid]
                keep  = 1.0 if (FINAL_REVISION and epoch == self.epochs) else chrom.keep_fraction(epoch-1, self.epochs, self.rng)

                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                xb_k, yb_k, dropped = self._drop_batch(xb, yb, keep)

                out = self.model(xb_k)
                loss = ce(out, yb_k)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                S = per_chrom[cid]
                S["loss_sum"]   += float(loss.item())
                S["batches"]    += 1
                S["samps_kept"] += int(yb_k.size(0))
                S["dropped"]    += int(dropped)
                total_samples_bp += int(yb_k.size(0))

                preds = out.argmax(1)
                run_correct += int((preds == yb_k).sum().item())
                bsz = int(yb_k.size(0))
                run_total += bsz
                run_loss_sum += float(loss.item()) * bsz

                if tqdm is not None:
                    iterator.set_postfix(keep=f"{keep:.2f}", loss=f"{loss.item():.4f}")

            fitness = []
            for cid in range(self.vpop):
                S = per_chrom[cid]
                fitness.append(S["loss_sum"] / max(1, S["batches"]))

            val_loss, val_acc = self._eval_loss_acc(self.val_loader)
            val_acc_hist.append(val_acc)

            order = sorted(range(self.vpop), key=lambda i: fitness[i])
            survivors_cnt = max(1, int(round(self.vpop * surv_frac)))
            survivors_idx = order[:survivors_cnt]
            survivors = [self.pop[i] for i in survivors_idx]

            children: List[Chromosome] = []
            for _ in range(self.vpop - len(survivors)):
                i = self.rng.randrange(self.vpop)
                j = self.rng.randrange(self.vpop)
                child = _recombine(self.pop[i], self.pop[j], self.rng)
                child = _mutate_es(child, epoch, self.epochs, self.rng)
                children.append(child)

            self.pop = survivors + children

            # hard-converge in the last K epochs (guarantees 100% homogeneity)
            if HARD_CONVERGE_LAST_K > 0 and epoch >= self.epochs - HARD_CONVERGE_LAST_K + 1:
                best = self.pop[order[0] if order[0] < len(self.pop) else 0]
                self.pop = [Chromosome(best.beta, best.noise_level, best.sigma_beta, best.sigma_noise)
                            for _ in range(self.vpop)]

            if self.rng.random() < reshuffle_p:
                self.rng.shuffle(self.assign)

            # reporting
            sig = {}
            for c in self.pop:
                key = (round(c.beta, 2), round(c.noise_level, 2))
                sig[key] = sig.get(key, 0) + 1
            maj_key, maj_cnt = max(sig.items(), key=lambda kv: kv[1])
            homo_pct = 100.0 * maj_cnt / self.vpop

            kept_total = sum(S["samps_kept"] for S in per_chrom)
            train_loss = 0.0 if run_total == 0 else run_loss_sum / run_total
            train_acc  = 0.0 if run_total == 0 else run_correct / run_total
            epoch_train_acc.append(round(train_acc, 4))
            samples_used_per_epoch.append(int(kept_total))
            time_per_epoch.append(time.time() - t0)

            print(f"\n=== Epoch {epoch} Report ===")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val  Loss: {val_loss:.4f} |  Val  Acc: {val_acc:.4f}")
            print(f"Samples Kept (for backprop): {kept_total}")
            print(f"Homogeneity: {homo_pct:5.2f}%  (majority ~ beta≈{maj_key[0]:.2f}, noise={maj_key[1]:.2f})")
            header = " cid |  beta | noise | keep(e) | batches | samps_kept | dropped | mean_CE "
            print(header); print("-" * len(header))
            for cid in range(self.vpop):
                c = self.pop[cid]
                S = per_chrom[cid]
                keep_e = 1.0 if (FINAL_REVISION and epoch == self.epochs) else c.keep_fraction(epoch-1, self.epochs, self.rng)
                mean_ce = S["loss_sum"] / max(1, S["batches"])
                print(f"{cid:4d} | {c.beta:5.2f} | {c.noise_level:5.2f} | {keep_e:6.2f} | "
                      f"{S['batches']:7d} | {S['samps_kept']:10d} | {S['dropped']:7d} | {mean_ce:7.4f}")
            print("=" * len(header))

        test_loss, test_acc = self._eval_loss_acc(self.test_loader)
        print(f"\n=== FINAL TEST ===  loss={test_loss:.4f} acc={test_acc:.4f}")

        # plots
        plot_accuracy_time_multi(self.model_name + "_genetic_train",
                                 epoch_train_acc, time_per_epoch, self.save_path, self.save_path)
        plot_accuracy_time_multi_test(self.model_name + "_genetic_val",
                                      val_acc_hist, time_per_epoch, samples_used_per_epoch,
                                      KEEP_FLOOR, self.save_path, self.save_path)

        return self.model, total_samples_bp
