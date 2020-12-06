from src.algorithms.ddm import DDM
from src.algorithms.eddm import EDDM
from src.algorithms.hddma import HDDM_A
from src.algorithms.hddmw import HDDM_W
from src.algorithms.precision import Precision
from src.loading.data_loader import DataLoader
from src.plotting.plot_prec_and_drift import plot
from src.prediction.predict import get_classifier, get_classifier_vfdt, get_classifier_bayes, get_classifier_perceptron


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
        clf.partial_fit([one_row[0]], [one_row[1]])
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
    clf = get_classifier(clf_data[0], clf_data[1])
    algorithms = [Precision(), HDDM_W()]
    process(algorithms, clf, dl)
    plot(algorithms)


def runVFDTwithDDM():
    dl = DataLoader()
    clf = get_classifier_vfdt()
    algorithms = [Precision(), DDM()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runVFDTwithEDDM():
    dl = DataLoader()
    clf = get_classifier_vfdt()
    algorithms = [Precision(), EDDM()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runVFDTwithHDDM_A():
    dl = DataLoader()
    clf = get_classifier_vfdt()
    algorithms = [Precision(), HDDM_A()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runVFDTwithHDDM_W():
    dl = DataLoader()
    clf = get_classifier_vfdt()
    algorithms = [Precision(), HDDM_W()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runBayeswithDDM():
    dl = DataLoader()
    clf = get_classifier_bayes()
    algorithms = [Precision(), DDM()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runBayeswithEDDM():
    dl = DataLoader()
    clf = get_classifier_bayes()
    algorithms = [Precision(), EDDM()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runBayeswithHDDM_A():
    dl = DataLoader()
    clf = get_classifier_bayes()
    algorithms = [Precision(), HDDM_A()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runBayeswithHDDM_W():
    dl = DataLoader()
    clf = get_classifier_bayes()
    algorithms = [Precision(), HDDM_W()]
    process_inc(algorithms, clf, dl)
    plot(algorithms)


def runClassificationWithInc(algorithm, classifier):
    dl = DataLoader()
    algorithms = [Precision(), algorithm]
    process_inc(algorithms, classifier, dl)
    plot(algorithms)


# runDDM()
# runEDDM()
# runHDDM_A()
# runHDDM_W()

algorithms = [DDM(),
              EDDM(),
              HDDM_A(),
              HDDM_W()]
classifiers = [
    get_classifier_vfdt(),
    get_classifier_bayes()
]

for alg in algorithms:
    runClassificationWithInc(alg, get_classifier_perceptron())
