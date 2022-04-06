import matplotlib.pyplot as plt
from matplotlib import style

class Visualize:
    def __init__(self, epochs_list, *args):
        self.epochs = epochs_list
        self.afl = args[0]  # av_fitness_list
        self.asl = args[2]  #av_score_list
        self.sl = args[3]   #score_list
        self.fl = args[1]   #fitness_list

    def visualize(self):
        style.use('ggplot')

        for ind, e in enumerate(self.epochs):
            max_fitness = max(self.fl[ind])
            plt.scatter([e for _ in range(1000)], self.fl[ind], color='purple', label='All fitnesses per epoch')
            plt.plot([e], [max_fitness], color='red',label='Max fitness per epoch')

        plt.plot(self.epochs, self.afl, marker=".", label='Average fitness per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        for ind, e in enumerate(self.epochs):
            max_score = max(self.sl[ind])
            plt.scatter([e for _ in range(1000)], self.sl[ind], color='purple', label='All scores per epoch')
            plt.plot([e], [max_score], color='red',label='Max score per epoch')

        plt.plot(self.epochs, self.asl, marker=".", label='Average score per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

