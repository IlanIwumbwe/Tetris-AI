import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import os

input_size = 7
output_size = 1
weights_min = -1
weights_max = 1

device = 'cpu'

class LinearNet(nn.Module):
    def __init__(self, output_w=None):
        super(LinearNet, self).__init__()
        if not output_w:
            self.fc2 = nn.Linear(input_size, output_size, bias=False).to(device)
            self.fc2.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.fc2.weight, weights_min, weights_max)
        else:
            self.fc2 = output_w

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# genetic algorithm
mutation_prob = 0.3
elitism = 0.2
mutation_power= 0.4

class Population:
    def __init__(self, size=150, old_population=None):
        self.size = size
        self.fitnesses = np.zeros(self.size)
        # at the start of the game, there's no old population
        if old_population is None:
            # set up population of nueral nets
            self.models = [LinearNet() for _ in range(size)]
        else:
            # get all nueral nets from previous iteration
            self.old_models = old_population.models
            self.old_fitnesses = old_population.fitnesses
            # setup models for this iteration, will fill list in mutation and crossover phase
            self.models = []
            self.crossover()
            self.mutate()


    def crossover(self):
        print('Crossover Process')
        # setup higher probabilities for higher performing neural nets
        sum_of_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i]/sum_of_fitnesses for i in range(self.size)]

        # sort by order of fitness (descending)
        fitness_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size*elitism:
                model_c = self.old_models[fitness_indices[i]]
            else:
                # select 2 best performing fitness indices using probs
                a, b = np.random.choice(self.size, size=2, p=probs, replace=False)

                # setup models from those indices
                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = LinearNet()

                prob_from_a = 0.5

                for j in range(input_size):
                    # perform cross-over process
                    if np.random.random() > prob_from_a:
                        model_c.fc2.weight.data[0][j] = model_b.fc2.weight.data[0][j]
                    else:
                        model_c.fc2.weight.data[0][j] = model_a.fc2.weight.data[0][j]

            # add new object to population
            self.models.append(model_c)

    def mutate(self):
        print('Mutation Process')
        for model in self.models:
            for i in range(input_size):
                if np.random.random() > mutation_prob:
                    noise = torch.randn(1).mul(mutation_power).to(device)
                    model.fc2.weight.data[0][i].add(noise[0])



