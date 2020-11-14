from src.algorithms.ddm import DDM
from src.algorithms.eddm import EDDM
from src.loading.data_loader import DataLoader
from src.prediction.predict import get_classifier


def process(algorithm):
    dl = DataLoader()
    clf_data = dl.get_predict_data()
    clf = get_classifier(clf_data[0], clf_data[1])
    # print()
    one_row = True
    while True:
        one_row = dl.get_data()
        if one_row is None:
            break
        prediction = clf.predict([one_row[0]])[0]
        if prediction == one_row[1]:
            algorithm.add_data(1, str(one_row[2]))
        else:
            algorithm.add_data(0, str(one_row[2]))

algorithm = DDM()
# algorithm = EDDM()
process(algorithm)