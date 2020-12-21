from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.drift_detection.eddm import EDDM as EDDM2

class EDDM(AlgorithmTemplate):
    def __init__(self):
        self.eddm = EDDM2()
        self.eddm.FDDM_MIN_NUM_INSTANCES = 60
        self.eddm.FDDM_WARNING = 0.95
        self.eddm.FDDM_OUTCONTROL = 0.92
        self.warnings = 0
        self.changes = 0
        self.last_info = None
        self.gathered_data = []
        self.name = "EDDM"

    def add_data(self, data, info):
        self.eddm.add_element(data)
        if self.last_info is None:
            self.last_info = info
        if self.last_info != info:
            self.last_info = info
            self.gathered_data.append([self.warnings, self.changes])
            self.warnings = 0
            self.changes = 0
        if self.eddm.detected_warning_zone():
            # print('Warning zone has been detected in year: ' + info)
            self.warnings += 1
        if self.eddm.detected_change():
            # print('Change has been detected in year: ' + info)
            self.changes += 1

        return None
