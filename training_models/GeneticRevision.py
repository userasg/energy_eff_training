import math
import random
import uuid
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch import optim
from selective_gradient import TrainRevision
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test

# Hyperparameters for GA and annealing (unchanged)
MAX_ALPHA            = 10.0
MAX_BETA             = 10.0
INIT_ELITISM         = 0.2
FINAL_ELITISM        = 0.8
INIT_MUTATION_RATE   = 0.4
FINAL_MUTATION_RATE  = 0.05
MUTATION_STD         = 0.1
PENALTY_COEFF        = 1.0  # λ in fitness = loss + λ*(kept/original)

# Fixed, simple quantization to allow real homogeneity and curve caching
_GRID_STEP = 0.5   # snap α, β to multiples of 0.5

class GeneticRevision(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader,
                 device, epochs, save_path, threshold, seed=42,
                 pop_size: int = None):
        """
        pop_size: number of chromosomes (policies). If None, defaults to min(128, #batches).
        """
        super().__init__(model_name, model, train_loader, test_loader,
                         device, epochs, save_path, threshold)
        self.rng  = random.Random(seed)
        nbatches  = len(train_loader)
        self.population_size = min(128, nbatches) if pop_size is None else min(pop_size, nbatches)
        self.schedules = ['power','exponential','logarithmic','inverse_linear','sigmoid_complement']
        self._x = np.arange(1, self.epochs + 1, dtype=np.float32)

        # cache for (schedule, alpha, beta) -> keep_frac curve
        self.curve_cache = {}

        self._init_population()

        # we keep this map to avoid key errors / track new uids; no heavy logic
        self.uid_history = {c['uid']: [] for c in self.population}

        # mapping from batch index -> chromosome index (built each epoch)
        self.batch2chrom = None

    # ---------- helpers ----------
    def _snap(self, v: float, vmin: float, vmax: float) -> float:
        """Snap to 0.5 grid so policies can be exactly identical."""
        snapped = round(v / _GRID_STEP) * _GRID_STEP
        return float(max(vmin, min(vmax, snapped)))

    def _base_curve(self, sched: str, alpha: float) -> np.ndarray:
        """
        Return a monotonically-decreasing curve in (0,1], length==epochs.
        NOTE: sigmoid_complement fixed to decrease with epoch.
        """
        x = self._x
        if sched == 'power':            # 1 / x^α
            y = 1.0 / (x ** alpha + 1e-8)
        elif sched == 'exponential':    # exp(-α x)
            y = np.exp(-alpha * x)
        elif sched == 'logarithmic':    # 1 / log(x+α)
            y = 1.0 / np.log(x + alpha + 1e-8)
        elif sched == 'inverse_linear': # 1 / (x+α)
            y = 1.0 / (x + alpha)
        else:                           # 1 - sigmoid(αx)
            y = 1.0 - 1.0 / (1.0 + np.exp(-alpha * x))
        y = y / (y.max() + 1e-12)
        return y.astype(np.float32)

    def _get_keep_frac(self, sched: str, alpha: float, beta: float) -> np.ndarray:
        """
        Cached per-epoch keep FRACTION in [1e-3, 1], key'd by exact (sched, α, β).
        """
        key = (sched, alpha, beta)
        cached = self.curve_cache.get(key)
        if cached is not None:
            return cached
        base = self._base_curve(sched, alpha)                 # (0,1]
        frac = np.clip(base * (beta / MAX_BETA), 1e-3, 1.0)   # scale by β, keep ≤ 1
        self.curve_cache[key] = frac
        return frac

    # ---------- Population init ----------
    def _init_population(self):
        """
        Initialize chromosomes + precompute KEEP FRACTION curves.
        α, β are snapped to 0.5 grid -> identical policies are possible (homogeneity > 0%).
        """
        self.population = []
        for _ in range(self.population_size):
            uid   = uuid.uuid4().hex
            sched = self.rng.choice(self.schedules)
            # snap to grid
            alpha = self._snap(self.rng.uniform(1.0, MAX_ALPHA), 1.0, MAX_ALPHA)
            beta  = self._snap(self.rng.uniform(1.0, MAX_BETA),  1.0, MAX_BETA)

            keep_frac = self._get_keep_frac(sched, alpha, beta)

            self.population.append({
                'uid':         uid,
                'schedule':    sched,
                'alpha':       alpha,       # already snapped
                'beta':        beta,        # already snapped
                'keep_frac':   keep_frac,   # per-epoch fraction (cached)
            })

    # ---------- Batch assignment (contiguous blocks to cut overhead) ----------
    def _assign_chromosomes_to_batches(self):
        """
        Assign contiguous blocks of batches to chromosomes.
        Each chromosome owns ~ceil(#batches / pop_size) consecutive batches.
        This reduces per-batch randomness and Python overhead.
        """
        nb = len(self.train_loader)
        P  = self.population_size
        block = math.ceil(nb / P)

        assignment = []
        order = list(range(P))
        self.rng.shuffle(order)  # reshuffle policy order each epoch for fairness

        for idx in order:
            assignment.extend([idx] * block)
            if len(assignment) >= nb:
                break
        self.batch2chrom = assignment[:nb]

    # ---------- Dropout application ----------
    def _apply_dropout_schedule(self, inputs, labels, chrom, epoch):
        bs = inputs.size(0)
        frac = float(chrom['keep_frac'][epoch])    # 0..1
        k = max(1, min(bs, int(math.ceil(frac * bs))))
        idx = torch.randperm(bs, device=inputs.device)[:k]
        return inputs[idx], labels[idx]

    # ---------- Evolution ----------
    def _evolve_population(self, fitness_avg, epoch):
        """
        Anneal exploration and make one policy dominate fast:
        - Elitism & mutation rate anneal as before.
        - Mutation jitter anneals via elitism (no extra hyperparams).
        - Clone the single best policy to 2*n_elite (exact duplicates -> rising homogeneity).
        - Tournament selection pressure tied to elitism.
        """
        frac          = epoch / (self.epochs - 1) if self.epochs > 1 else 1.0
        elitism       = INIT_ELITISM + frac*(FINAL_ELITISM - INIT_ELITISM)
        mutation_rate = INIT_MUTATION_RATE + frac*(FINAL_MUTATION_RATE - INIT_MUTATION_RATE)
        eff_mut_std   = MUTATION_STD * (1.0 - elitism)

        # Rank by averaged fitness (lower is better)
        sorted_pop = sorted(self.population, key=lambda c: fitness_avg[c['uid']])
        n_elite    = max(1, int(self.population_size * elitism))
        elites     = sorted_pop[:n_elite]

        new_pop = [dict(e) for e in elites]

        # Aggressive elite cloning up to 2*n_elite
        best = elites[0]
        clone_target = min(self.population_size, 2 * n_elite)
        while len(new_pop) < clone_target:
            clone = dict(best)
            clone['uid'] = uuid.uuid4().hex
            self.uid_history.setdefault(clone['uid'], [])
            new_pop.append(clone)

        # Tournament pressure grows with elitism
        k_tourn = max(2, int(2 + 8 * elitism))
        def pick():
            pool = self.rng.sample(elites, min(k_tourn, len(elites)))
            return min(pool, key=lambda c: fitness_avg[c['uid']])

        # Fill the rest
        while len(new_pop) < self.population_size:
            p1, p2 = pick(), pick()
            ca = 0.5*(p1['alpha'] + p2['alpha'])
            cb = 0.5*(p1['beta']  + p2['beta'])
            cs = self.rng.choice([p1['schedule'], p2['schedule']])

            # mutate schedule with annealed rate
            if self.rng.random() < mutation_rate:
                cs = self.rng.choice(self.schedules)
            # mutate params with annealed std
            if self.rng.random() < mutation_rate:
                ca += self.rng.gauss(0, eff_mut_std)
                cb += self.rng.gauss(0, eff_mut_std)

            # snap to grid & clip
            ca = self._snap(ca, 1.0, MAX_ALPHA)
            cb = self._snap(cb, 1.0, MAX_BETA)

            child = {'uid': uuid.uuid4().hex, 'schedule': cs,
                     'alpha': ca, 'beta': cb}

            child['keep_frac'] = self._get_keep_frac(cs, ca, cb)

            self.uid_history.setdefault(child['uid'], [])
            new_pop.append(child)

        for c in new_pop:
            self.uid_history.setdefault(c['uid'], [])

        self.population = new_pop

    # ---------- Training ----------
    def train_with_genetic(self):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_acc, epoch_loss = [], []
        test_acc, test_loss   = [], []
        times, samples_used   = [], []
        total_steps = 0
        t0 = time.time()

        nbatches = len(self.train_loader)

        for e in range(self.epochs):
            self._assign_chromosomes_to_batches()
            self.model.train()

            rl = correct = kept_total = 0
            fit_sum = {c['uid']: 0.0 for c in self.population}
            fit_cnt = {c['uid']: 0   for c in self.population}

            for i, (x, y) in enumerate(tqdm(self.train_loader, desc=f"GA Epoch {e+1}")):
                chrom = self.population[self.batch2chrom[i]]
                inp, lab = x.to(self.device), y.to(self.device)
                inp_s, lab_s = self._apply_dropout_schedule(inp, lab, chrom, e)

                out  = self.model(inp_s)
                loss = criterion(out, lab_s)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                lval = float(loss.detach().cpu().item())
                k    = lab_s.size(0)
                rl   += lval
                correct    += (out.argmax(1) == lab_s).sum().item()
                kept_total += k
                total_steps += k

                obj = lval + PENALTY_COEFF * (k / x.size(0))
                fit_sum[chrom['uid']] += obj
                fit_cnt[chrom['uid']] += 1

            epoch_loss.append(rl / nbatches)
            epoch_acc.append(correct / kept_total if kept_total else 0.0)
            times.append(time.time() - t0)
            samples_used.append(kept_total)

            fitness_avg = {uid: (fit_sum[uid] / max(1, fit_cnt[uid])) for uid in fit_sum}

            # evolve
            self._evolve_population(fitness_avg, e)

            # homogeneity AFTER evolution
            counts = {}
            for c in self.population:
                key = (c['schedule'], c['alpha'], c['beta'])  # exact matches possible thanks to snapping
                counts[key] = counts.get(key, 0) + 1
            top_key = max(counts, key=counts.get)
            homo_pct = counts[top_key] / self.population_size * 100.0
            print(f"  → Population homogeneity (post-evolve): {homo_pct:.1f}% | mode={top_key}")

            # evaluation
            self.model.eval()
            tl = tc = tt = 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    l   = criterion(out, y)
                    tl += l.item()
                    tc += (out.argmax(1) == y).sum().item()
                    tt += y.size(0)

            test_loss.append(tl / len(self.test_loader))
            test_acc.append(tc / tt)

            print(
                f"Epoch {e+1}/{self.epochs}   "
                f"Train Loss: {epoch_loss[-1]:.4f}, Acc {epoch_acc[-1]:.4f}   |   "
                f"Test Loss:  {test_loss[-1]:.4f}, Acc {test_acc[-1]:.4f}"
            )

            scheduler.step(test_loss[-1])

        log_memory(t0, time.time())
        plot_accuracy_time_multi(
            self.model_name + "_genetic", epoch_acc, times,
            self.save_path, self.save_path
        )
        plot_accuracy_time_multi_test(
            self.model_name + "_genetic", test_acc, times, samples_used,
            self.threshold, self.save_path, self.save_path
        )

        return self.model, total_steps
