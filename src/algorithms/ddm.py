from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.drift_detection import DDM as DDM2

class DDM(AlgorithmTemplate):
    def __init__(self):
        self.ddm = DDM2(warning_level=2.2, out_control_level=2.4, min_num_instances=60)
        self.warnings = 0
        self.changes = 0
        self.last_info = None
        self.gathered_data = []
        self.name = "DDM"

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
