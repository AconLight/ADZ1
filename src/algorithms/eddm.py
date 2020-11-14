from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.drift_detection.eddm import EDDM as EDDM2

class EDDM(AlgorithmTemplate):
    def __init__(self):
        self.eddm = EDDM2()
        self.eddm.FDDM_MIN_NUM_INSTANCES = 60
        self.eddm.FDDM_WARNING = 0.95
        self.eddm.FDDM_OUTCONTROL = 0.92

    def add_data(self, data, info):
        self.eddm.add_element(data)
        # print("processed" + str(data))
        if self.eddm.detected_warning_zone():
            print('Warning zone has been detected in year: ' + info)
        if self.eddm.detected_change():
            print('Change has been detected in year: ' + info)

        return None
