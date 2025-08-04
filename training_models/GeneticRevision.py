import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from tqdm import tqdm
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from selective_gradient import TrainRevision  # Inherit from this


class GeneticRevision(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path, threshold, population_size=10):
        super().__init__(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold)
        self.population_size = population_size
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            genome = {
                'decay_type': np.random.choice(['power', 'exp', 'log', 'inv_lin', 'sigmoid']),
                'alpha': np.random.uniform(0.1, 3.0)
            }
            population.append(genome)
        return population

    def evaluate_fitness(self, genome, data_size, start_revision):
        decay_type = genome['decay_type']
        alpha = genome['alpha']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        self.model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for epoch in range(self.epochs):
            sample_ratio = self.compute_schedule(epoch + 1, data_size, alpha, decay_type) / data_size
            sample_ratio = min(1.0, sample_ratio)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                selected_count = max(1, int(sample_ratio * batch_size))

                indices = torch.randperm(batch_size)[:selected_count]
                inputs_selected = inputs[indices]
                labels_selected = labels[indices]

                optimizer.zero_grad()
                outputs = self.model(inputs_selected)
                loss = criterion(outputs, labels_selected)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                with torch.no_grad():
                    preds = torch.argmax(self.model(inputs), dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy  # Can also combine with -total_loss for multi-objective

    def compute_schedule(self, epoch, data_size, alpha, decay_type):
        if decay_type == 'power':
            return self.power_law_decay(epoch, data_size, alpha)
        elif decay_type == 'exp':
            return self.exponential_decay(epoch, data_size, alpha)
        elif decay_type == 'log':
            return self.log_schedule(epoch, data_size, alpha)
        elif decay_type == 'inv_lin':
            return self.inverse_linear(epoch, alpha)
        elif decay_type == 'sigmoid':
            return self.sigmoid_complement_decay(epoch, data_size, alpha)

    def evolve_population(self, data_size, start_revision):
        fitness_scores = [self.evaluate_fitness(g, data_size, start_revision) for g in self.population]
        sorted_idx = np.argsort(fitness_scores)[::-1]
        top_genomes = [self.population[i] for i in sorted_idx[:self.population_size // 2]]

        new_population = top_genomes.copy()
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(top_genomes, 2, replace=False)
            child = {
                'decay_type': np.random.choice([parent1['decay_type'], parent2['decay_type']]),
                'alpha': np.clip(np.mean([parent1['alpha'], parent2['alpha']]) + np.random.normal(0, 0.1), 0.1, 3.0)
            }
            new_population.append(child)

        self.population = new_population

    def train_genetic_scheduler(self, generations, data_size, start_revision):
        for gen in range(generations):
            print(f"\nGeneration {gen+1}/{generations}")
            self.evolve_population(data_size, start_revision)
            best_genome = max(self.population, key=lambda g: self.evaluate_fitness(g, data_size, start_revision))
            print(f"Best Genome: {best_genome}")

        return best_genome  # Can be used in final training phase

    #### Dropout schedule math functions inherited ####
    def inverse_linear(self, epoch, alpha):
        x = np.arange(1, 200)
        y = 1 / (x + alpha)
        y_scaled = (y / np.max(y)) * 50000
        return y_scaled[epoch - 1]

    def log_schedule(self, epoch, data_size, alpha):
        x = np.arange(1, 200)
        y = 1 / np.log(x + alpha)
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]

    def power_law_decay(self, epoch, data_size, alpha):
        x = np.arange(1, self.epochs + 1)
        y = 1 / (x ** alpha)
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]

    def exponential_decay(self, epoch, data_size, alpha):
        x = np.arange(1, self.epochs + 1)
        y = np.exp(-alpha * x)
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]

    def sigmoid_complement_decay(self, epoch, data_size, alpha):
        x = np.arange(1, self.epochs + 1)
        y = 1 - 1 / (1 + np.exp(-alpha * x))
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]
