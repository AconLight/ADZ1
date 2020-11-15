from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.trees import HoeffdingTree


class Precision(AlgorithmTemplate):
    def __init__(self):
        self.correct = 0
        self.count = 0
        self.correctness_dist = []
        self.last_info = None
        self.gathered_data = []
        self.name = "precision"

    def add_data(self, data, info):
        self.correctness_dist.append(data)
        self.correct += data
        self.count += 1
        if self.last_info is None:
            self.last_info = info
        if self.last_info != info:
            self.last_info = info
            calc_prec = self.count_correctness()
            self.gathered_data.append(calc_prec)
            # print("precision for year " + info + ": " + str(calc_prec))
            self.correctness_dist = []


    def count_correctness(self):
        cor = 0
        al = len(self.correctness_dist)
        for val in self.correctness_dist:
            cor += val

        return cor*1.0/al




