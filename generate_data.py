import numpy as np
import math


def generate_data(N=100):
    theta = np.linspace(0, 2 * math.pi, N) + math.pi * (np.random.rand(N)) / 100
    a = 0.5
    b = 0.5
    r1 = 0.4 + 2*(np.random.rand(N)-0.5)/10
    x1 = a + r1*np.cos(theta) + (np.random.rand(N)-0.5)/50
    y1 = b + r1*np.sin(theta) + (np.random.rand(N)-0.5)/50
    r2 = 0.2*np.random.rand(N)
    x2 = a + r2*np.cos(theta) + (np.random.rand(N)-0.5)/50
    y2 = b + r2*np.sin(theta) + (np.random.rand(N)-0.5)/50

    samples1 = [[x1[i], y1[i]] for i in range(len(x1))]
    samples2 = [[x2[i], y2[i]] for i in range(len(x2))]
    sample_label = {}
    for i in range(len(samples1)):
        sample1 = samples1[i]
        sample2 = samples2[i]
        # one hot
        sample_label[(sample1[0], sample1[1])] = [1, 0]
        sample_label[(sample2[0], sample2[1])] = [0, 1]

    samples = samples1 + samples2
    random_samples = np.random.permutation(samples)

    random_labels = np.array([sample_label[(sample[0], sample[1])] for sample in random_samples])

    # print('random_labels', random_labels)
    # print('random_samples', random_samples)

    return random_samples, random_labels


def load_partial(samples, labels, step):
    index = 0
    while index < len(samples):
        yield samples[index: index + step], labels[index: index + step]
        index += step
    return


if __name__ == '__main__':
    generate_data(100)
    # print(generate_data(50))
    # print(np.linspace(0, 2 * math.pi, 10))
    # print(np.random.rand(50))

