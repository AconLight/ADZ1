

def get_classifier(x, y):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(x, y)

    return clf


def get_classifier_vfdt():
    from skmultiflow.trees import HoeffdingTree

    clf = HoeffdingTree(split_confidence=0.5)

    return clf