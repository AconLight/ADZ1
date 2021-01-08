from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
from skmultiflow.trees import HoeffdingTree
import numpy as np


class IncrementalClassifier:
    def __init__(self):
        pass

    def partial_fit(self, one_row):
        pass

    def predict(self, x):
        pass


class VFDT(IncrementalClassifier):
    def __init__(self, grace_period=200, split_confidence=0.5, leaf_prediction='nba', split_criterion='info_gain'):
        super().__init__()
        self.clf = HoeffdingTree(split_confidence=split_confidence,
                                 grace_period=grace_period,
                                 leaf_prediction=leaf_prediction,
                                 split_criterion=split_criterion)

    def partial_fit(self, one_row):
        self.clf.partial_fit([one_row[0]], [one_row[1]])

    def predict(self, x):
        return self.clf.predict(x)


class Bayes(IncrementalClassifier):

    def __init__(self):
        super().__init__()
        self.clf = NaiveBayes()

    def partial_fit(self, one_row):
        self.clf.partial_fit([one_row[0]], [one_row[1]])

    def predict(self, x):
        return self.clf.predict(x)


class AEEC():
    def __init__(self, beta=0.8, gamma=0.1, pruning='weakest'):
        super().__init__()
        self.clf = AdditiveExpertEnsembleClassifier(beta=beta,
                                                    gamma=gamma,
                                                    pruning=pruning)

    def partial_fit(self, one_row):
        self.clf.fit_single_sample([one_row[0]], [one_row[1]])

    def predict(self, x):
        return self.clf.predict(x)

class DWM():
    def __init__(self, clfs, beta):
        super().__init__()
        self.clfs = clfs
        self.weights = np.ones(len(clfs))
        self.beta = beta
        self.prediction = 0
        self.predictions = []

    def partial_fit(self, one_row):
        self.dwm_clfs_fit(one_row)

    def predict(self, x):
        return self.dwm_predict(x)

    def dwm_clfs_fit(self, one_row):
        for i in range(len(self.clfs)):
            self.clfs[i].partial_fit(one_row)
            if self.predictions[i] != one_row[1]:
                self.weights[i] *= self.beta

    def dwm_predict(self, x):
        self.predictions = []
        for clf in self.clfs:
            self.predictions.append(clf.predict(x))

        self.prediction = 0
        sum_weights = 0
        for w in self.weights:
            sum_weights += w

        for i in range(len(self.predictions)):
            self.prediction += self.predictions[i]*self.weights[i]

        self.prediction /= sum_weights
        if self.prediction > 0.5:
            self.prediction = 1
        else:
            self.prediction = 0

        return [self.prediction]

