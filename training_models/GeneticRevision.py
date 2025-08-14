# GeneticRevision.py — GA random-dropout with equal partitions, 2dp math (except share%), and 0.1-step params
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from utils import plot_accuracy_time_multi, plot_accuracy_time_multi_test

# ----------------------------- few knobs (trimmed) -----------------------------
POP_SIZE            = 16
KEEP_FLOOR          = 0.4
WARMUP_EPOCHS       = 1
MAX_DECAY_PER_E     = 0.25
LR                  = 2e-4

ELITE_FRAC          = 0.20
MUT_RATE_BASE       = 0.50
MUT_STD             = 0.40
REPLACE_START       = 0.45
REPLACE_END         = 0.85
EXPLOIT_FRAC        = 0.75
CLONE_ELITE_PROB    = 0.50

SELECTION_STRATEGY  = "tournament"    # "roulette" or "tournament"
TOURNAMENT_K        = 3

# fitness blend (kept internal): more weight on accuracy
_W_ACC, _W_LOSS, _W_EFF = 0.75, 0.15, 0.10

# ----- alpha/gamma ranges + step quantization -----
PARAM_STEP          = 0.10
ALPHA_INIT_RANGE     = (0.70, 1.30)
GAMMA_INIT_RANGE     = (0.90, 1.40)
ALPHA_CLAMP          = (0.10, 3.00)
GAMMA_CLAMP          = (0.50, 2.50)

# =========================== tiny schedule object ===========================
class ChromosomeSchedule:
    """
    kind ∈ {"power","invlog"}
    alpha: shape  ∈ [0.10, 5.00], snapped to 0.10 steps (2dp)
    gamma: sharp  ∈ [0.10, 5.00], snapped to 0.10 steps (2dp)
    share: display-only fraction (exact from partition; NOT forced to 2dp)
    """
    @staticmethod
    def _snap_grid(x: float, lo: float, hi: float, step: float) -> float:
        # clamp, then snap to the nearest step; keep 2dp representation
        x = max(lo, min(hi, float(x)))
        q = round(x / step) * step
        return float(round(max(lo, min(hi, q)), 2))

    def __init__(self, kind, alpha, gamma, share):
        self.kind  = kind
        self.alpha = ChromosomeSchedule._snap_grid(alpha, ALPHA_CLAMP[0], ALPHA_CLAMP[1], PARAM_STEP)
        self.gamma = ChromosomeSchedule._snap_grid(gamma, GAMMA_CLAMP[0], GAMMA_CLAMP[1], PARAM_STEP)
        self.share = float(share)  # exact proportion from partition (can be non-2dp)

    def _raw(self, e: int) -> float:
        e = max(1, int(e))
        if self.kind == "power":
            return 1.0 / (e ** self.alpha)
        else:  # "invlog" : log arg > 1.0 since e>=1 and alpha>=0.1 → e+alpha>=1.1
            return 1.0 / math.log(e + self.alpha)

    def keep_frac(self, e: int, keep_floor: float) -> float:
        """
        keep(1)=1.00, monotone → keep_floor:
        keep_e = keep_floor + (1-keep_floor) * (r(e)/r(1))**gamma
        """
        r1 = self._raw(1)
        re = self._raw(e)
        ratio = 0.0 if r1 <= 0 else (re / r1)
        keep = keep_floor + (1.0 - keep_floor) * (ratio ** self.gamma)
        return float(round(min(1.0, max(keep_floor, keep)), 2))

    def drop_frac(self, e: int, keep_floor: float) -> float:
        return float(round(1.0 - self.keep_frac(e, keep_floor), 2))


# =========================== data partition helper ===========================
class _IndexedView(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

class DataPartitions:
    """Disjoint fixed partitions + monotone random shrinking per chromosome."""
    def __init__(self, base_loader, population, rng):
        self.base_loader = base_loader
        self.population  = population
        self._rng        = rng
        self._build()

    def _build(self):
        ds = self.base_loader.dataset
        N  = len(ds)
        idx = list(range(N))
        self._rng.shuffle(idx)

        # exact equal integer split (difference ≤ 1) — fairness > 2dp rule here
        sizes = self._equal_split(N, POP_SIZE)

        self.owner = {}         # dataset index -> chrom_id
        self.orig  = []         # list[int] per chromosome
        self.active= []         # active subset per chromosome (shrinks)
        start = 0
        for ci, sz in enumerate(sizes):
            seg = idx[start:start+sz]; start += sz
            for j in seg: self.owner[j] = ci
            self.orig.append(seg)
            self.active.append(list(seg))
            # set display share from actual partition size (exact ratio; NOT forced to 2dp)
            self.population[ci].share = sz / float(N)

        # cache loader kwargs
        bl = self.base_loader
        self._loader_kw = dict(
            batch_size=getattr(bl, "batch_size", 128),
            num_workers=getattr(bl, "num_workers", 0),
            pin_memory=getattr(bl, "pin_memory", False),
            drop_last=getattr(bl, "drop_last", False),
            collate_fn=getattr(bl, "collate_fn", None),
        )

    @staticmethod
    def _equal_split(N, k):
        base = N // k
        sizes = [base] * k
        for i in range(N - base * k):  # first remainder partitions get +1
            sizes[i] += 1
        return sizes

    def make_epoch_loader(self, union_indices):
        base = _IndexedView(self.base_loader.dataset)
        kw = {k: v for k, v in self._loader_kw.items() if v is not None}
        return DataLoader(base, sampler=SubsetRandomSampler(union_indices), shuffle=False, **kw)

    def update_actives(self, epoch, keep_floor, warmup_epochs, max_decay_per_e, schedules):
        """
        Warmup: 100% keep; afterward smooth decay:
        keep_rel_new = min(last, max(schedule_keep, last*(1 - MAX_DECAY_PER_E))).
        Returns {chrom_id: newly_dropped_this_epoch}
        """
        dropped = {}
        for ci, chrom in enumerate(schedules):
            last_keep = len(self.active[ci]) / max(1, len(self.orig[ci]))
            if epoch <= warmup_epochs:
                target_keep_rel = 1.00
            else:
                e_sched = epoch - warmup_epochs + 1
                sched_keep = chrom.keep_frac(e_sched, keep_floor)
                lower_bound = round(max(keep_floor, last_keep * (1.0 - max_decay_per_e)), 2)
                target_keep_rel = min(last_keep, max(sched_keep, lower_bound))

            target_count = max(1, int(round(target_keep_rel * len(self.orig[ci]))))
            cur = self.active[ci]
            if len(cur) > target_count:
                self._rng.shuffle(cur)
                self.active[ci] = cur[:target_count]
                dropped[ci] = len(cur) - target_count
            else:
                dropped[ci] = 0
        return dropped

    def union_indices(self):
        return [i for seg in self.active for i in seg]


# ============================== metrics helper ==============================
class Metrics:
    def __init__(self, pop_size): self.reset(pop_size)
    def reset(self, pop_size):
        self.loss_sum = {i: 0.0 for i in range(pop_size)}
        self.correct  = {i: 0   for i in range(pop_size)}
        self.count    = {i: 0   for i in range(pop_size)}
        self.kept_ep  = {i: 0   for i in range(pop_size)}

    def add_batch(self, idxs, loss_each, corr_vec, owner_map):
        for r in range(len(idxs)):
            ci = owner_map[int(idxs[r])]
            self.count[ci]   += 1
            self.kept_ep[ci] += 1
            self.loss_sum[ci] += round(float(loss_each[r]), 2)  # 2dp here
            self.correct[ci]  += int(corr_vec[r])

    def per_chrom_means(self):
        means = []
        for i in range(len(self.count)):
            c = max(1, self.count[i])
            ce = round(self.loss_sum[i]/c, 2)
            acc = round(self.correct[i]/c, 2)
            means.append((ce, acc))
        return means

    def fitness(self, _weights_ignored):
        n = len(self.count)
        losses = [round(self.loss_sum[i] / max(1, self.count[i]), 2) for i in range(n)]
        accs   = [round(self.correct[i]  / max(1, self.count[i]), 2) for i in range(n)]
        kept   = [float(self.kept_ep[i]) for i in range(n)]

        def _minmax(xs):
            lo, hi = min(xs), max(xs)
            if round(hi - lo, 2) == 0.00:
                return [0.50 for _ in xs]
            return [round((x - lo) / (hi - lo), 2) for x in xs]

        inv = lambda xs: [round(1.0 - v, 2) for v in _minmax(xs)]
        acc_s, loss_s, eff_s = _minmax(accs), inv(losses), inv(kept)
        return [round(_W_ACC*acc_s[i] + _W_LOSS*loss_s[i] + _W_EFF*eff_s[i], 2) for i in range(n)]


# ============================== the trainer (GA) ==============================
class GeneticRevision:
    """
    - Partition train set once into POP_SIZE equal slices (fixed owners).
    - Each epoch: random-drop within each slice to hit a schedule (power or invlog),
      with warmup and per-epoch smoothness cap.
    - GA evolves (kind, alpha, gamma) snapped to 0.10 steps; population homogenises late.
    - Evaluates: TRAIN (online), VALIDATION (per epoch), TEST (final).
    """
    def __init__(self, model_name, model, train_loader, val_loader, test_loader,
                 device, epochs, save_path=None, seed=42):
        self.model_name = model_name
        self.model      = model
        self.device     = device
        self.epochs     = int(epochs)
        self.save_path  = save_path
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.rng = random.Random(seed)
        random.seed(seed); torch.manual_seed(seed)

        # initialise schedules (alpha/gamma from widened ranges; snapped to 0.10 grid)
        self.population = []
        for _ in range(POP_SIZE):
            kind  = "power" if self.rng.random() < 0.5 else "invlog"
            alpha = ChromosomeSchedule._snap_grid(
                self.rng.uniform(*ALPHA_INIT_RANGE), ALPHA_CLAMP[0], ALPHA_CLAMP[1], PARAM_STEP
            )
            gamma = ChromosomeSchedule._snap_grid(
                self.rng.uniform(*GAMMA_INIT_RANGE), GAMMA_CLAMP[0], GAMMA_CLAMP[1], PARAM_STEP
            )
            self.population.append(ChromosomeSchedule(kind, alpha, gamma, 1.0/POP_SIZE))

        # partitions over the *train* dataset (sets exact shares per partition)
        self.parts = DataPartitions(self.train_loader, self.population, self.rng)
        self._epoch = 0

    # ------------------------------- public entry -------------------------------
    def train_with_genetic(self):
        dev = self.device
        self.model.to(dev)
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        lr_sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

        train_acc_hist, val_acc_hist, times, samples_used = [], [], [], []
        total_steps = 0

        for epoch in range(1, self.epochs+1):
            self._epoch = epoch
            t0 = time.time()

            dropped_epoch = self.parts.update_actives(
                epoch, KEEP_FLOOR, WARMUP_EPOCHS, MAX_DECAY_PER_E, self.population
            )

            union_idx = self.parts.union_indices()
            epoch_loader = self.parts.make_epoch_loader(union_idx)

            # -------- train one epoch --------
            self.model.train()
            M = Metrics(len(self.population))
            seen, corr_total, loss_sum_total = 0, 0, 0.0

            for x, y, idxs in tqdm(epoch_loader, desc=f"[GA] Epoch {epoch}/{self.epochs}", leave=False):
                x, y = x.to(dev), y.to(dev)
                logits = self.model(x)
                loss_each = F.cross_entropy(logits, y, reduction="none")
                loss = loss_each.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                preds = logits.argmax(1)
                correct_vec = (preds == y).to(torch.int)

                bs = y.size(0)
                seen += bs
                corr_total += int(correct_vec.sum().item())
                loss_sum_total += round(float(loss_each.sum().detach().cpu()), 2)

                M.add_batch(idxs, loss_each.detach().cpu().tolist(),
                            correct_vec.detach().cpu().tolist(), self.parts.owner)

            train_loss = round(loss_sum_total / max(1, seen), 2)
            train_acc  = round(corr_total / max(1, seen), 2)
            train_acc_hist.append(train_acc)
            samples_used.append(seen)
            total_steps += seen

            # -------- validation --------
            val_loss, val_acc = self._eval_loss_acc(self.val_loader)
            lr_sched.step(val_loss)
            val_acc_hist.append(val_acc)

            # -------- GA update --------
            fitness = M.fitness((_W_ACC, _W_LOSS, _W_EFF))
            self._evolve(fitness)

            # -------- report --------
            self._print_epoch_report(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                kept_total=seen,
                dropped_epoch=dropped_epoch,
                fitness=fitness,
                per_means=M.per_chrom_means()
            )

            times.append(time.time() - t0)

        # -------- final TEST --------
        test_loss, test_acc = self._eval_loss_acc(self.test_loader)
        print(f"\n=== Final TEST ===  Loss: {test_loss:.2f} | Acc: {test_acc:.2f}\n")

        plot_accuracy_time_multi(self.model_name + "_genetic_train", train_acc_hist, times,
                                 self.save_path, self.save_path)
        plot_accuracy_time_multi_test(self.model_name + "_genetic_val", val_acc_hist, times, samples_used,
                                      KEEP_FLOOR, self.save_path, self.save_path)

        return self.model, total_steps

    # --------------------------- evolution internals ---------------------------
    def _current_mut_rate(self):
        frac_left = max(0.0, 1.0 - self._epoch / max(1, self.epochs))
        return round(max(0.05, MUT_RATE_BASE * (0.5 + 0.5 * frac_left)), 2)

    def _evolve(self, fitness):
        pop = self.population
        m = len(pop)
        frac_done = self._epoch / max(1, self.epochs)
        repl_rate = REPLACE_START + (REPLACE_END - REPLACE_START) * frac_done
        g = max(2, int(math.ceil(repl_rate * m)))

        order = sorted(range(m), key=lambda i: fitness[i], reverse=True)
        elite_n = max(1, int(round(ELITE_FRAC * m)))
        elites = [pop[i] for i in order[:elite_n]]

        children = []
        for _ in range(g):
            if frac_done >= EXPLOIT_FRAC and self.rng.random() < CLONE_ELITE_PROB:
                base = elites[0]
                child = ChromosomeSchedule(base.kind, base.alpha, base.gamma, 0.0)
                child = self._mutate(child, tiny=True)
            else:
                p1 = pop[self._select_parent(fitness)]
                p2 = pop[self._select_parent(fitness)]
                child = self._mutate(self._crossover(p1, p2))
            children.append(child)

        survivors = [pop[i] for i in order[elite_n:]]
        survivors = survivors[:-g] if g < len(survivors) else []

        new_pop = elites + survivors + children
        if len(new_pop) > m: new_pop = new_pop[:m]
        while len(new_pop) < m: new_pop.append(self._mutate(self.rng.choice(elites)))

        # keep display shares aligned with existing ones
        for i in range(m): new_pop[i].share = pop[i].share
        self.population = new_pop

    def _select_parent(self, fitness):
        if SELECTION_STRATEGY == "tournament":
            k = max(2, TOURNAMENT_K)
            cand = self.rng.sample(range(len(fitness)), k=k)
            return max(cand, key=lambda i: fitness[i])
        # roulette
        base = min(fitness)
        shifted = [round(f - base, 2) for f in fitness]
        total = sum(shifted)
        if total <= 0: return self.rng.randrange(len(fitness))
        r, c = self.rng.random() * total, 0.0
        for i, f in enumerate(shifted):
            c += f
            if r <= c: return i
        return len(fitness) - 1

    def _crossover(self, a, b):
        kind  = a.kind if self.rng.random() < 0.5 else b.kind
        w1, w2 = self.rng.random(), self.rng.random()
        alpha  = w1 * a.alpha + (1.0 - w1) * b.alpha
        gamma  = w2 * a.gamma + (1.0 - w2) * b.gamma
        # snap to grid + clamp + 2dp
        alpha  = ChromosomeSchedule._snap_grid(alpha, ALPHA_CLAMP[0], ALPHA_CLAMP[1], PARAM_STEP)
        gamma  = ChromosomeSchedule._snap_grid(gamma, GAMMA_CLAMP[0], GAMMA_CLAMP[1], PARAM_STEP)
        return ChromosomeSchedule(kind, alpha, gamma, 0.0)

    def _mutate(self, c, tiny=False):
        out = ChromosomeSchedule(c.kind, c.alpha, c.gamma, 0.0)
        mut = self._current_mut_rate() * (0.25 if tiny else 1.0)
        if self.rng.random() < round(mut * 0.25, 2):
            out.kind = "power" if c.kind == "invlog" else "invlog"
        if self.rng.random() < mut:
            a = out.alpha + self.rng.gauss(0.0, MUT_STD * (0.5 if tiny else 1.0))
            out.alpha = ChromosomeSchedule._snap_grid(a, ALPHA_CLAMP[0], ALPHA_CLAMP[1], PARAM_STEP)
        if self.rng.random() < mut:
            g = out.gamma + self.rng.gauss(0.0, MUT_STD * (0.5 if tiny else 1.0))
            out.gamma = ChromosomeSchedule._snap_grid(g, GAMMA_CLAMP[0], GAMMA_CLAMP[1], PARAM_STEP)
        return out

    # ------------------------------- evaluation -------------------------------
    @torch.no_grad()
    def _eval_loss_acc(self, loader):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        crit = nn.CrossEntropyLoss(reduction="sum")
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2: continue
            x, y = batch[:2]
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss_sum += crit(logits, y).item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
        mean_loss = 0.0 if total == 0 else round(loss_sum / total, 2)
        acc = 0.0 if total == 0 else round(correct / total, 2)
        self.model.train()
        return mean_loss, acc

    # ------------------------------- reporting --------------------------------
    def _print_epoch_report(self, epoch, train_loss, train_acc, val_loss, val_acc,
                            kept_total, dropped_epoch, fitness, per_means):
        # homogeneity
        buckets = {}
        for c in self.population:
            key = (c.kind, round(c.alpha, 2), round(c.gamma, 2))
            buckets[key] = buckets.get(key, 0) + 1
        majority = max(buckets, key=lambda k: buckets[k])
        homo_pct = round(100.0 * buckets[majority] / len(self.population), 2)

        total_N = sum(len(seg) for seg in self.parts.orig)  # exact share denominator

        print(f"\n=== Epoch {epoch} Report ===")
        print(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f}")
        print(f" Val  Loss: {val_loss:.2f} |  Val  Acc: {val_acc:.2f}")
        print(f"Samples Kept (for backprop): {kept_total}")
        print(f"Homogeneity: {homo_pct:5.2f}%  (majority ~ {majority[0]}, alpha≈{majority[1]:.2f}, gamma≈{majority[2]:.2f})")

        header = " idx | kind   | share% | alpha  | gamma  | drop(e) | kept_ep | dropped | mean_CE | train_acc | fitness "
        print(header); print("-"*len(header))
        for i, c in enumerate(self.population):
            share_pct = 100.0 * len(self.parts.orig[i]) / max(1, total_N)  # exact ratio; not forced to 2dp globally
            drop_e = c.drop_frac(max(1, epoch - WARMUP_EPOCHS + 1), KEEP_FLOOR) if epoch > 1 else 0.00
            mean_ce, mean_tracc = per_means[i]
            print(f" {i:3d} | {c.kind:6s} | {share_pct:6.2f} | {c.alpha:6.2f} | {c.gamma:6.2f} | "
                  f"{drop_e:7.2f} | {self.parts.active[i].__len__():7d} | {dropped_epoch.get(i,0):7d} | "
                  f"{mean_ce:7.2f} | {mean_tracc:9.2f} | {fitness[i]:7.2f}")
        print("="*len(header))
