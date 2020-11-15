import matplotlib.pyplot as plt

def plot(algorithms):
    for algorithm in algorithms:
        plt.plot(range(1995, 1995+len(algorithm.gathered_data)), algorithm.gathered_data)
        plt.ylabel(algorithm.name)
        plt.show()