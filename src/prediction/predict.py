def get_classifier(x, y):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(x, y)

    return clf


def get_classifier_vfdt():
    from skmultiflow.trees import HoeffdingTree

    return HoeffdingTree(split_confidence=0.5)


def get_classifier_bayes():
    from skmultiflow.bayes import NaiveBayes

    return NaiveBayes()


def get_classifier_perceptron():
    from skmultiflow.neural_networks import PerceptronMask

    return PerceptronMask()
