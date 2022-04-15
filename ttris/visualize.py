import matplotlib.pyplot as plt
from matplotlib import style

class Visualize:
    def __init__(self, epochs_list, av_fitnesses, all_fitnesses, av_scores, all_scores):
        self.epochs = epochs_list
        self.afl = av_fitnesses
        self.asl = av_scores
        self.sl = all_scores
        self.fl = all_fitnesses

    def visualize(self):
        style.use('ggplot')

        max_fitnesses = []
        for ind, e in enumerate(self.epochs):
            max_fitnesses.append(max(self.fl[ind]))
            plt.scatter([e for _ in range(1000)], self.fl[ind], color='lightskyblue')

        plt.plot(self.epochs, max_fitnesses, color='red',label='Max fitness per epoch',marker='.')

        plt.plot(self.epochs, self.afl, marker=".", label='Average fitness per epoch',color='gold')
        plt.xlabel('Epochs')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        plt.plot(self.epochs, self.afl, marker=".", label='Average fitness per epoch', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        max_scores = []
        for ind, e in enumerate(self.epochs):
            max_scores.append(max(self.sl[ind]))
            plt.scatter([e for _ in range(1000)], self.sl[ind], color='lightseagreen')

        plt.plot(self.epochs, max_scores, color='red',label='Max score per epoch',marker='.')

        plt.plot(self.epochs, self.asl, marker=".", label='Average score per epoch',color='gold')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

        plt.plot(self.epochs, self.asl, marker=".", label='Average score per epoch', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
