from src.algorithms.ddm import DDM
from src.algorithms.eddm import EDDM
from src.algorithms.hddma import HDDM_A
from src.algorithms.hddmw import HDDM_W
from src.algorithms.precision import Precision
from src.loading.data_loader import DataLoader
from src.plotting.plot_prec_and_drift import plot
from src.prediction.inc_predict import VFDT, Bayes, AEEC, DWM
from src.prediction.predict import get_classifier


def process(algorithms, clf, dl):
    one_row = True
    while True:
        one_row = dl.get_data()
        if one_row is None:
            break
        prediction = clf.predict([one_row[0]])[0]
        if prediction == one_row[1]:
            for algorithm in algorithms:
                algorithm.add_data(1, str(one_row[2]))
        else:
            for algorithm in algorithms:
                algorithm.add_data(0, str(one_row[2]))


def process_inc(algorithms, clf, dl):
    one_row = True
    while True:
        one_row = dl.get_data()
        if one_row is None:
            break
        if one_row[1] == 'Algiers':
            one_row[1] = 0
        else:
            one_row[1] = 1
        prediction = clf.predict([one_row[0]])[0]
        clf.partial_fit(one_row)
        if prediction == one_row[1]:
            for algorithm in algorithms:
                algorithm.add_data(1, str(one_row[2]))
        else:
            for algorithm in algorithms:
                algorithm.add_data(0, str(one_row[2]))


def runDDM():
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    clf = get_classifier(clf_data[0], clf_data[1])
    algorithms = [Precision(), DDM()]
    process(algorithms, clf, dl)
    plot(algorithms)


def runEDDM():
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    clf = get_classifier(clf_data[0], clf_data[1])
    algorithms = [Precision(), EDDM()]
    process(algorithms, clf, dl)
    plot(algorithms)


def runHDDM_A():
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    clf = get_classifier(clf_data[0], clf_data[1])
    algorithms = [Precision(), HDDM_A()]
    process(algorithms, clf, dl)
    plot(algorithms)


def runHDDM_W():
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    algorithms = [Precision(), HDDM_W()]
    process(algorithms, clf, dl)
    plot(algorithms)


def run_classification(algorithm, classifier):
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    process_inc(algorithms, classifier, dl)
    plot(algorithms)


def run_classification_with_inc(algorithm, classifier):
    dl = DataLoader()
    algorithms = [Precision(), algorithm]
    process_inc(algorithms, classifier, dl)
    plot(algorithms)

def run_classification_with_all(algorithm, classifiers):
    dl = DataLoader()
    algorithms = [Precision(), algorithm]
    process_inc(algorithms, classifier, dl)
    plot(algorithms)


# runDDM()
# runEDDM()
# runHDDM_A()
# runHDDM_W()

algorithms = [
    DDM(),
    EDDM(),
    HDDM_A(),
    HDDM_W()
]

# INCREMENTATION
classifiers_inc = [
    DWM([
        VFDT(split_criterion='gini'),
        VFDT(split_criterion='hellinger'),
        VFDT(split_criterion='gini'),
        VFDT(leaf_prediction='nba'),
        VFDT(leaf_prediction='nb'),
        VFDT(leaf_prediction='mc'),
        VFDT(grace_period=100),
        VFDT(grace_period=200),
        Bayes(), #no arguments
        AEEC(pruning='oldest'),
        AEEC(pruning='weakest'),
        AEEC(beta=0.8),
        AEEC(beta=0.9),
        AEEC(gamma=0.1),
        AEEC(gamma=0.2)
    ], 0.9)
]

for clf in classifiers_inc:
    run_classification_with_inc(HDDM_A(), clf)
