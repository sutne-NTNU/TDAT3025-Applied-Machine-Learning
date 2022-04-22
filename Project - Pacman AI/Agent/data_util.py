import os
from pathlib import Path

import matplotlib.pylab as plt
import torch

PATH = os.path.join(os.path.dirname(__file__), '')

learning_values = PATH + 'data/learning_values.txt'
evaluation_values = PATH + 'data/evaluation_values.txt'
network_path, network_suffix = PATH + 'networks/', '.pt'


def saveStateDict(stateDict, filename):
    torch.save(stateDict, network_path + filename + network_suffix)


def loadStateDict(filename):
    my_file = Path(network_path + filename + network_suffix)
    if my_file.is_file():
        return torch.load(network_path + filename + network_suffix)
    else:
        return None


def plot_evaluation_values(plot_exact=True,
                           plot_averages=True,
                           averageOverEpisodes=50,
                           saveFig=True):
    """Plot the scores of each episode from the training, with the option to save the plot"""
    values, x, y = [], [], []
    file = open(evaluation_values)
    for line in file:
        val = line.strip().split(',')
        x.append(int(val[0]))
        y.append(int(val[1]))

    if plot_exact:
        plt.plot(x, y, '.', color='0.75', label="Score")

    if plot_averages:
        averages = []
        for i, value in enumerate(y):
            sumInterval = 0
            for j in range(max(0, i + 1 - averageOverEpisodes), i + 1):
                sumInterval += y[j]
            averages.append(sumInterval / min(i + 1, averageOverEpisodes))
        plt.plot(x, averages, '-', color='b', label="Trend")

    plt.legend()
    plt.title('Score under evaluering etter x antall episoder med trening')
    plt.xlabel('Episoder med trening')
    plt.ylabel('Total score')
    plt.savefig('./media/evaluation.png') if saveFig else None
    plt.show()


def plot_learning_values(plot_exact=True,
                         plot_averages=True,
                         averageOverEpisodes=200,
                         saveFig=True):
    values1, values2 = [], []
    file = open(learning_values)
    for line in file:
        values = line.strip().split(',')
        values1.append(int(values[0]))
        values2.append(int(values[1]))

    if plot_exact:
        plt.plot(values1, '.', color='0.65', label="Steg")
        plt.plot(values2, '.', color='0.8', label="Score")

    if plot_averages:
        average1 = []
        for i, value in enumerate(values1):
            sumInterval = 0
            for j in range(max(0, i + 1 - averageOverEpisodes), i + 1):
                sumInterval += values1[j]
            average1.append(sumInterval / min(i + 1, averageOverEpisodes))
        average2 = []
        for i, value in enumerate(values2):
            sumInterval = 0
            for j in range(max(0, i + 1 - averageOverEpisodes), i + 1):
                sumInterval += values2[j]
            average2.append(sumInterval / min(i + 1, averageOverEpisodes))

        plt.plot(average1, '-', color='r', label="Trend for antall steg")
        plt.plot(average2, '-', color='b', label="Trend for total Score")

    plt.legend()
    plt.title('Utvikling av total score og antall steg per episode under trening')
    plt.xlabel('Episode')
    plt.savefig('./media/learning.png') if saveFig else None
    plt.show()


def plot_average():
    scores = []
    file = open(evaluation_values)
    for line in file:
        val = line.strip().split(',')
        scores.append(int(val[1]))

    plt.plot(scores, '.', color='0.65', label="Score")
    average = sum(scores) / len(scores)
    plt.plot([0, len(scores) - 1], [average, average], '-', color='r', label="Gjennomsnittlig Score: %i" % average)
    plt.legend()
    plt.title('100% Tilfeldige handlinger')
    plt.xlabel('Episode')
    plt.savefig('./media/random_scores.png')
    plt.show()


def getLineCount(filename):
    my_file = Path(filename)
    if not my_file.is_file():
        return 0
    line_count = 0
    file = open(learning_values)
    for line in file:
        line_count += 1
    return line_count


def save_to_file(file, values: list):
    file = open(file, 'a+')
    for value in values:
        if len(value) > 1:
            # step_count , score
            # episodes_trained , score
            file.write('%i,%i\n' % (value[0], value[1]))
        else:
            file.write("%i\n" % value)


if __name__ == "__main__":
    plot_learning_values(saveFig=False)
    plot_evaluation_values(saveFig=False)
