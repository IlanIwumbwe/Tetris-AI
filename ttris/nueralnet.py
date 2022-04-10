import numpy as np
import scipy.special
import pickle

class Nueral_net:
    def __init__(self, input_nodes, hidden_a_nodes, hidden_b_nodes, output_nodes):
        self.inodes = input_nodes
        self.hnodes_a = hidden_a_nodes
        self.hnodes_b = hidden_b_nodes
        self.onodes = output_nodes

        self.activation_func = lambda x : scipy.special.expit(x)

        # weight matrix between input and hidden
        self.wi_ha = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes_a, self.inodes))
        self.wha_hb = np.random.normal(0.0, pow(self.hnodes_a, -0.5), (self.hnodes_b, self.hnodes_a))
        self.whb_o = np.random.normal(0.0, pow(self.hnodes_b, -0.5), (self.onodes, self.hnodes_b))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_a_inputs = np.matmul(self.wi_ha, inputs)
        hidden_a_outputs = self.activation_func(hidden_a_inputs)
        hidden_b_inputs = np.matmul(self.wha_hb, hidden_a_outputs)
        hidden_b_outputs = self.activation_func(hidden_b_inputs)

        final_inputs = np.matmul(self.whb_o, hidden_b_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

hidden_a_size = 8
hidden_b_size = 8
output_size = 1

# genetic algorithm
mutation_prob = 0.08
elitism = 0.2
mutation_power= 0.1

class Population:
    def __init__(self, size, old_population, input_size):
        self.size = size
        self.fitnesses = np.zeros(self.size)
        self.input_size = input_size
        # at the start of the game, there's no old population
        if old_population is None:
            # set up population of nueral nets
            self.models = [Nueral_net(self.input_size, hidden_a_size, hidden_b_size, output_size) for _ in range(self.size)]
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
                child = self.old_models[fitness_indices[i]]
            else:
                # select 2 best performing fitness indices using probs, do not replace,
                # so indices cannot be selected twice
                a, b = np.random.choice(self.size, size=2, p=probs, replace=False)

                # setup models from those indices
                parent_a, parent_b = self.old_models[a], self.old_models[b]
                child = Nueral_net(self.input_size, hidden_a_size, hidden_b_size, output_size)
                a_fitness, b_fitness = self.old_fitnesses[a], self.old_fitnesses[b]

                if a_fitness == 0 and b_fitness == 0:
                    child.wi_ha = np.zeros((hidden_a_size, self.input_size))
                    child.wha_hb = np.zeros((hidden_b_size, hidden_a_size))
                    child.whb_o = np.zeros((output_size, hidden_b_size))

                    prob_from_a = 0.5

                    # cross-over weights between inputs and hidden a
                    for row_ind, row in enumerate(child.wi_ha):
                        for col_ind, col in enumerate(row):
                            if np.random.random() < prob_from_a:
                                child.wi_ha[row_ind][col_ind] = parent_a.wi_ha[row_ind][col_ind]
                            else:
                                child.wi_ha[row_ind][col_ind] = parent_b.wi_ha[row_ind][col_ind]

                    # cross-over weights between hidden a and hidden b
                    for row_ind, row in enumerate(child.wha_hb):
                        for col_ind, col in enumerate(row):
                            if np.random.random() < prob_from_a:
                                child.wha_hb[row_ind][col_ind] = parent_a.wha_hb[row_ind][col_ind]
                            else:
                                child.wha_hb[row_ind][col_ind] = parent_b.wha_hb[row_ind][col_ind]

                    # cross-over weights between hidden b and output
                    for row_ind, row in enumerate(child.whb_o):
                        for col_ind, col in enumerate(row):
                            if np.random.random() < prob_from_a:
                                child.whb_o[row_ind][col_ind] = parent_a.whb_o[row_ind][col_ind]
                            else:
                                child.whb_o[row_ind][col_ind] = parent_b.whb_o[row_ind][col_ind]

                else:
                    child.wi_ha = (a_fitness/(a_fitness+b_fitness))*parent_a.wi_ha + (b_fitness/(a_fitness+b_fitness))*parent_b.wi_ha
                    child.wha_hb = (a_fitness/(a_fitness+b_fitness))*parent_a.wha_hb + (b_fitness/(a_fitness+b_fitness))*parent_b.wha_hb
                    child.whb_o = (a_fitness/(a_fitness+b_fitness))*parent_a.whb_o + (b_fitness/(a_fitness+b_fitness))*parent_b.whb_o

            # add new object to population
            self.models.append(child)

    def mutate(self):
        print('Mutation Process')
        for model in self.models:
            for row in model.wi_ha:
                for ind, col in enumerate(row):
                    if np.random.random() < mutation_prob:
                        row[ind] += np.random.uniform(-mutation_power, mutation_power)

            for row in model.wha_hb:
                for ind, col in enumerate(row):
                    if np.random.random() < mutation_prob:
                        row[ind] += np.random.uniform(-mutation_power, mutation_power)

            for row in model.whb_o:
                for ind, col in enumerate(row):
                    if np.random.random() < mutation_prob:
                        row[ind] += np.random.uniform(-mutation_power, mutation_power)

    def save_population(self, epoch_number):
        weight_matrices = []
        fitnesses = []

        for model in self.models:
            weight_matrices.append((model.wi_ha, model.wha_hb, model.whb_o))

        for fitness_list in self.fitnesses:
            fitnesses.append(fitness_list)

        path1 = f"./populations/{epoch_number+1}population.pkl"
        with open(path1, "wb") as f:
            pickle.dump(weight_matrices, f)

        path2 = f"./populations/{epoch_number+1}fitness.pkl"
        with open(path2, "wb") as f:
            pickle.dump(fitnesses, f)









