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


class GeneticRevision(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path,
                 threshold, seed=42):
        super().__init__(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold)
        self.seed = seed
        self.rng = random.Random(seed)
        # one chromosome per batch
        self.population_size = len(train_loader)
        # available schedule types (must match TrainRevision methods)
        self.schedules = [
            'power', 'exponential', 'logarithmic', 'inverse_linear', 'sigmoid_complement'
        ]
        # initialize population and history of batch assignments
        self._init_population()
        self.uid_history = {chrom['uid']: [] for chrom in self.population}

    def _init_population(self):
        """Populate initial chromosomes with unique UID, random schedule and alpha."""
        self.population = []
        for _ in range(self.population_size):
            uid = uuid.uuid4().hex
            schedule = self.rng.choice(self.schedules)
            alpha = round(self.rng.uniform(0.1, 3.0), 2)
            self.population.append({'uid': uid, 'schedule': schedule, 'alpha': alpha})

    def _shuffle_population(self):
        """Shuffle chromosome-to-batch mapping, avoiding assignment to recent batches."""
        indices = list(range(self.population_size))
        # retry until no chromosome is placed in its last 5 batch slots
        while True:
            perm = self.rng.sample(indices, k=self.population_size)
            if all(
                batch_idx not in self.uid_history[self.population[chrom_idx]['uid']]
                for batch_idx, chrom_idx in enumerate(perm)
            ):
                break
        # record new history and store shuffled mapping
        for batch_idx, chrom_idx in enumerate(perm):
            uid = self.population[chrom_idx]['uid']
            history = self.uid_history.setdefault(uid, [])
            history.append(batch_idx)
            if len(history) > 5:
                history.pop(0)
        self.shuffled_indices = perm

    def _apply_dropout_schedule(self, inputs, labels, chrom, epoch):
        """Select subset of samples according to schedule function from TrainRevision."""
        batch_size = inputs.size(0)
        # determine number of samples to keep based on schedule
        alpha = chrom['alpha']
        step = epoch + 1  # TrainRevision schedules are 1-based
        schedule = chrom['schedule']
        if schedule == 'power':
            keep = self.power_law_decay(step, batch_size, alpha)
        elif schedule == 'exponential':
            keep = self.exponential_decay(step, batch_size, alpha)
        elif schedule == 'logarithmic':
            keep = self.log_schedule(step, batch_size, alpha)
        elif schedule == 'inverse_linear':
            keep = self.inverse_linear(step, batch_size, alpha)
        elif schedule == 'sigmoid_complement':
            keep = self.sigmoid_complement_decay(step, batch_size, alpha)
        else:
            keep = batch_size
        # clamp and convert to int
        k = max(1, min(batch_size, int(keep)))
        # randomly sample k examples
        perm = torch.randperm(batch_size, device=inputs.device)
        idx = perm[:k]
        return inputs[idx], labels[idx]

    def _evolve_population(self, fitness):
        """Select elites and replace the rest with new random chromosomes."""
        # sort by loss (lower is better)
        sorted_pop = sorted(self.population, key=lambda c: fitness.get(c['uid'], float('inf')))
        num_elite = self.population_size // 2
        elites = sorted_pop[:num_elite]
        new_pop = elites.copy()
        # generate fresh chromosomes for bottom half
        for _ in range(self.population_size - num_elite):
            uid = uuid.uuid4().hex
            schedule = self.rng.choice(self.schedules)
            alpha = round(self.rng.uniform(0.1, 3.0), 2)
            new_pop.append({'uid': uid, 'schedule': schedule, 'alpha': alpha})
            self.uid_history[uid] = []
        self.population = new_pop

    def train_with_genetic(self):
        """Main training loop that applies GA-evolved dropout schedules per batch."""
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_accuracies, epoch_losses = [], []
        epoch_test_accuracies, epoch_test_losses = [], []
        time_per_epoch, samples_per_epoch = [], []
        num_steps = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            # remap chromosomes to batches
            self._shuffle_population()

            # -- training --
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            fitness = {}
            for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc=f"Genetic Epoch {epoch+1}")):
                chrom = self.population[self.shuffled_indices[batch_idx]]
                inputs, labels = x.to(self.device), y.to(self.device)
                inputs_sel, labels_sel = self._apply_dropout_schedule(inputs, labels, chrom, epoch)

                outputs = self.model(inputs_sel)
                loss = criterion(outputs, labels_sel)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_sel).sum().item()
                total += labels_sel.size(0)
                fitness[chrom['uid']] = loss.item()
                num_steps += labels_sel.size(0)

            # record per-epoch stats
            epoch_losses.append(running_loss / len(self.train_loader))
            epoch_accuracies.append(correct / total if total else 0)
            time_per_epoch.append(time.time() - start_time)
            samples_per_epoch.append(num_steps)

            # -- evaluation --
            self.model.eval()
            test_loss, test_correct, test_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    tl = criterion(out, y)
                    test_loss += tl.item()
                    preds = out.argmax(dim=1)
                    test_correct += (preds == y).sum().item()
                    test_total += y.size(0)

            epoch_test_losses.append(test_loss / len(self.test_loader))
            epoch_test_accuracies.append(test_correct / test_total)

            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Loss {epoch_losses[-1]:.4f} Acc {epoch_accuracies[-1]:.4f} | "
                  f"Test Loss {epoch_test_losses[-1]:.4f} Acc {epoch_test_accuracies[-1]:.4f}")
            scheduler.step(epoch_test_losses[-1])

            # GA evolution
            self._evolve_population(fitness)

        # logging and plotting
        log_memory(start_time, time.time())
        plot_accuracy_time_multi(
            self.model_name + "_genetic",
            epoch_accuracies,
            time_per_epoch,
            self.save_path,
            self.save_path
        )
        plot_accuracy_time_multi_test(
            # now passing samples_per_epoch and threshold correctly:
            self.model_name + "_genetic",
            epoch_test_accuracies,
            time_per_epoch,
            samples_per_epoch,
            self.threshold,
            self.save_path,
            self.save_path
        )

        return self.model, num_steps
