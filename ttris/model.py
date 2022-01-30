import torch
import torch.nn as nn
import numpy as np
import os

input_size = 4

# genetic algorithm
mutation_prob = 0.3
elitism = 0.2
mutation_power= 0.4

class Population:
    def __init__(self, size, old_population):
        self.size = size
        self.fitnesses = np.zeros(self.size)
        # at the start of the game, there's no old population
        if old_population is None:
            # set up population of nueral nets
            self.models = [np.array([np.random.uniform(-1, 1) for _ in range(input_size)]) for _ in range(self.size)]
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
                # if model within elitism critical region, select it as is
                model_c = self.old_models[fitness_indices[i]]
            else:
                # select 2 best performing fitness indices using probs
                a, b = np.random.choice(self.size, size=2, p=probs, replace=False)

                # setup models from those indices
                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = np.array([None for _ in range(input_size)])

                for j in range(input_size):
                    model_c[j] = self.old_fitnesses[b]*model_b[j]+self.old_fitnesses[a]*model_a[j]

            # add new object to population
            self.models.append(model_c)

    def mutate(self):
        print('Mutation Process')
        for model in self.models:
            for i in range(input_size):
                if np.random.random() > mutation_prob:
                    noise = np.random.uniform(-0.2, 0.2)

                    model[i] += noise




