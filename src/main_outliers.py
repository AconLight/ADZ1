from src.loading.data_loader import DataLoader
from src.outliers_algorithms.outliers import SHESD
from src.plotting.plot_prec_and_drift import plot_outliers


def calc_prec(data):
    good = 0
    all = 0
    for vals in data:
        if vals[0] and vals[1]:
            good += 1
            all += 1
        if vals[0] and not vals[1]:
            all += 1
    if all is 0:
        return None
    return good/all

def calc_acc(data):
    good = 0
    all = 0
    for vals in data:
        if vals[0] == vals[1]:
            good += 1
        all += 1
    return good/all

def calc_recall(data):
    good = 0
    all = 0
    for vals in data:
        if vals[0] and vals[1]:
            good += 1
            all += 1
        if not vals[0] and not vals[1]:
            all += 1
    if all is 0:
        return None
    return good/all

def runSHESD(window_size=365*2, errors=[5, 10 ,15, 20, 25, 30, 35, 40], percentages=[0.1], max_anomalieses=[70]):
    dl = DataLoader()
    results = []
    precisions = []
    recalls = []
    accuracies = []
    for error in errors:
        for percentage in percentages:
            for max_anomalies in max_anomalieses:
                dl.restart_itr()
                dl.apply_outliers(percentage, error)
                shesd = SHESD(max_anomalies)
                result = []
                for year in range(5):
                    data = []
                    for i in range(window_size):
                        data.append(dl.get_data())

                    result += shesd.check(data)

                results.append("value added: " + str(error) + ", outliers percentage: " + str(percentage*100) + "%, max_anomalies per window(" + str(window_size) + "days): " + str(max_anomalies) + ", recall: " + str(calc_recall(result)) + ", precision: " + str(calc_prec(result)) + ", accuracy: " + str(calc_acc(result)))
                precisions.append(calc_prec(result))
                recalls.append(calc_recall(result))
                accuracies.append(calc_acc(result))

    print(results[0])
    plot_outliers(precisions, recalls, accuracies, errors)





runSHESD()
