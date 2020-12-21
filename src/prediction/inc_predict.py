from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
from skmultiflow.trees import HoeffdingTree


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
