from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.drift_detection import HDDM_W as DDM2

class HDDM_W(AlgorithmTemplate):
    def __init__(self):
        self.ddm = DDM2()
        self.warnings = 0
        self.changes = 0
        self.last_info = None
        self.gathered_data = []
        self.name = "HDDM_W"

    def add_data(self, data, info):
        self.ddm.add_element(data)
        if self.last_info is None:
            self.last_info = info
        if self.last_info != info:
            self.last_info = info
            self.gathered_data.append([self.warnings, self.changes])
            self.warnings = 0
            self.changes = 0
        if self.ddm.detected_warning_zone():
            # print('Warning zone has been detected in year: ' + info)
            self.warnings += 1
        if self.ddm.detected_change():
            # print('Change has been detected in year: ' + info)
            self.changes += 1

        return None
