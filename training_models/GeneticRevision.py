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

# Hyperparameters for GA and annealing
MAX_ALPHA = 10.0
MAX_BETA  = 10.0
INIT_ELITISM = 0.2        # start keeping top 20%
FINAL_ELITISM = 0.8       # end keeping top 80%
INIT_MUTATION_RATE = 0.4 # start with high mutation
FINAL_MUTATION_RATE = 0.05 # end with low mutation
MUTATION_STD   = 0.1     # Gaussian mutation std for alpha/beta

class GeneticRevision(TrainRevision):
    def __init__(self, model_name, model, train_loader, test_loader,
                 device, epochs, save_path, threshold, seed=42):
        super().__init__(model_name, model, train_loader, test_loader,
                         device, epochs, save_path, threshold)
        self.seed = seed
        self.rng = random.Random(seed)
        self.population_size = len(train_loader)
        self.schedules = ['power', 'exponential', 'logarithmic',
                          'inverse_linear', 'sigmoid_complement']
        self._init_population()
        self.uid_history = {c['uid']: [] for c in self.population}

    def _init_population(self):
        self.population = []
        for _ in range(self.population_size):
            uid      = uuid.uuid4().hex
            sched    = self.rng.choice(self.schedules)
            alpha    = round(self.rng.uniform(1.0, MAX_ALPHA), 2)
            beta     = round(self.rng.uniform(1.0, MAX_BETA), 2)
            self.population.append({'uid':uid,'schedule':sched,'alpha':alpha,'beta':beta})

    def _shuffle_population(self):
        # unchanged
        indices = list(range(self.population_size))
        while True:
            perm = self.rng.sample(indices, k=self.population_size)
            if all(batch_idx not in self.uid_history[self.population[c]['uid']] 
                   for batch_idx,c in enumerate(perm)): break
        for batch_idx,c in enumerate(perm):
            uid = self.population[c]['uid']
            hist = self.uid_history.setdefault(uid,[])
            hist.append(batch_idx)
            if len(hist)>5: hist.pop(0)
        self.shuffled_indices = perm

    def _apply_dropout_schedule(self, inputs, labels, chrom, epoch):
        # unchanged (calls super().schedule with alpha,beta)
        batch_size = inputs.size(0)
        a,b = chrom['alpha'], chrom['beta']
        step = epoch+1
        sched = chrom['schedule']
        # call new signatures (alpha,beta)
        if sched=='power':    base = super().power_law_decay(step,batch_size,a,b)
        elif sched=='exponential': base = super().exponential_decay(step,batch_size,a,b)
        elif sched=='logarithmic': base = super().log_schedule(step,batch_size,a,b)
        elif sched=='inverse_linear': base = super().inverse_linear(step,batch_size,a,b)
        elif sched=='sigmoid_complement': base = super().sigmoid_complement_decay(step,batch_size,a,b)
        else: base = batch_size
        keep = base * b
        k = max(1, min(batch_size, int(keep)))
        idx = torch.randperm(batch_size,device=inputs.device)[:k]
        return inputs[idx], labels[idx]

    def _evolve_population(self, fitness, epoch):
        # dynamic elitism and mutation rate
        frac = epoch/(self.epochs-1)
        elitism = INIT_ELITISM + (FINAL_ELITISM-INIT_ELITISM)*frac
        mutation_rate = INIT_MUTATION_RATE + (FINAL_MUTATION_RATE-INIT_MUTATION_RATE)*frac

        sorted_pop = sorted(self.population, key=lambda c: fitness.get(c['uid'],float('inf')))
        num_elite = max(1,int(self.population_size * elitism))
        elites = sorted_pop[:num_elite]
        new_pop = [dict(e) for e in elites]
        while len(new_pop)<self.population_size:
            p1,p2 = self.rng.sample(elites,2)
            # crossover
            ca = (p1['alpha']+p2['alpha'])/2
            cb = (p1['beta'] +p2['beta']) /2
            cs = self.rng.choice([p1['schedule'],p2['schedule']])
            # mutation
            if self.rng.random()<mutation_rate:
                cs = self.rng.choice(self.schedules)
            ca += self.rng.gauss(0,MUTATION_STD)
            cb += self.rng.gauss(0,MUTATION_STD)
            ca = round(min(max(ca,1.0),MAX_ALPHA),2)
            cb = round(min(max(cb,1.0),MAX_BETA),2)
            uid = uuid.uuid4().hex
            new_pop.append({'uid':uid,'schedule':cs,'alpha':ca,'beta':cb})
            self.uid_history[uid]=[]
        self.population = new_pop

    def train_with_genetic(self):
        self.model.to(self.device)
        crit = nn.CrossEntropyLoss()
        opt  = optim.AdamW(self.model.parameters(),lr=3e-4)
        sched= StepLR(opt,step_size=1,gamma=0.98)

        epoch_acc,epoch_loss=[],[]
        test_acc,test_loss=[],[]
        times,samples=[],[]
        total_steps=0
        t0=time.time()

        for e in range(self.epochs):
            self._shuffle_population()
            self.model.train()
            rl=correct=kt=0
            fitness={}
            for i,(x,y) in enumerate(tqdm(self.train_loader,desc=f"GA Epoch {e+1}")):
                chrom = self.population[self.shuffled_indices[i]]
                inp,lab = x.to(self.device),y.to(self.device)
                inp_s,lab_s = self._apply_dropout_schedule(inp,lab,chrom,e)
                out= self.model(inp_s)
                l = crit(out,lab_s)
                opt.zero_grad();l.backward();opt.step()
                rl+=l.item()
                p=out.argmax(1)
                correct += (p==lab_s).sum().item()
                kt+=lab_s.size(0)
                total_steps+=lab_s.size(0)
                fitness[chrom['uid']]=l.item()

            epoch_loss.append(rl/len(self.train_loader))
            epoch_acc.append(correct/kt if kt else 0)
            times.append(time.time()-t0)
            samples.append(kt)

            # eval
            self.model.eval()
            tl=tc=tt=0
            with torch.no_grad():
                for x,y in self.test_loader:
                    x,y=x.to(self.device),y.to(self.device)
                    out=self.model(x)
                    l=crit(out,y)
                    tl+=l.item()
                    p=out.argmax(1)
                    tc+=(p==y).sum().item()
                    tt+=y.size(0)
            test_loss.append(tl/len(self.test_loader))
            test_acc.append(tc/tt)

            print(f"Epoch {e+1}/{self.epochs}  ",
                  f"Train Loss: {epoch_loss[-1]:.4f}, Acc: {epoch_acc[-1]:.4f}  |  ",
                  f"Test Loss: {test_loss[-1]:.4f}, Acc: {test_acc[-1]:.4f}")

            # evolve with epoch info for annealing
            self._evolve_population(fitness, e)
            sched.step(test_loss[-1])

        log_memory(t0, time.time())
        plot_accuracy_time_multi(self.model_name+"_genetic", epoch_acc,times,
                                 self.save_path,self.save_path)
        plot_accuracy_time_multi_test(self.model_name+"_genetic", test_acc,times,
                                      samples,self.threshold,
                                      self.save_path,self.save_path)
        return self.model, total_steps
