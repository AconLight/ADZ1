from sklearn.ensemble import RandomForestClassifier


def get_classifier(x, y):
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(x, y)

    return clf
