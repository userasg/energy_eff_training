import random
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch import optim
from selective_gradient import TrainRevision
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test

class SimpleSwitcher(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path,
                 threshold, data_size, switch_interval=5, start_revision=0, seed=42):
        super().__init__(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold)
        
        self.data_size = data_size
        self.switch_interval = switch_interval
        self.start_revision = start_revision 
        self.seed = seed                    
        random.seed(self.seed)               
        
        self.available_schedules = [
            self.power_law_decay,
            self.exponential_decay,
            self.log_schedule,
            self.inverse_linear,
            self.sigmoid_complement_decay
        ]
        self.current_schedule = random.choice(self.available_schedules)
        self.revision_epochs = self._generate_random_revision_epochs()

    def _generate_random_revision_epochs(self):
        valid_range = list(range(self.start_revision, self.epochs))
        if not valid_range:
            print("⚠️ No valid epochs for revision (start_revision >= total epochs).")
            return []
        max_possible = len(valid_range)
        num_revisions = random.randint(1, max_possible)
        revision_epochs = sorted(random.sample(valid_range, num_revisions))
        print(f"[Seed={self.seed}] Revision will occur at epochs: {revision_epochs}")
        return revision_epochs


    def train_with_switching(self):
        save_path = self.save_path
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses, epoch_accuracies = [], []
        epoch_test_accuracies, epoch_test_losses = [], []
        time_per_epoch, samples_used_per_epoch = [], []
        num_step = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss, total_correct, total_samples = 0.0, 0, 0

            # Switch schedule
            if epoch % self.switch_interval == 0:
                self.current_schedule = random.choice(self.available_schedules)
                print(f"Switched to schedule: {self.current_schedule.__name__}")

            use_full_data = epoch in self.revision_epochs

            print(f"Epoch [{epoch+1}/{self.epochs}] {'[FULL REVISION]' if use_full_data else ''}")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if not use_full_data:
                    # Apply schedule
                    scaled_value = self.current_schedule(epoch + 1, self.data_size, alpha=2)
                    sample_ratio = scaled_value / self.data_size
                    selected_count = max(1, int(sample_ratio * inputs.size(0)))
                    selected_indices = torch.randperm(inputs.size(0))[:selected_count]
                    inputs = inputs[selected_indices]
                    labels = labels[selected_indices]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                num_step += inputs.size(0)
                samples_used += inputs.size(0)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)

                progress_bar.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # Evaluate
            self.model.eval()
            correct, total, test_loss = 0, 0, 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            val_loss = test_loss / len(self.test_loader)
            print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)

        plot_accuracy_time_multi(self.model_name, epoch_accuracies, time_per_epoch, save_path, save_path)
        plot_accuracy_time_multi_test(self.model_name, epoch_test_accuracies, time_per_epoch, samples_used_per_epoch, self.threshold, save_path, save_path)

        eff_epoch = int(num_step / self.data_size)
        return self.model, num_step
