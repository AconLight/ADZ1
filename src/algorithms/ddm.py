from src.algorithm_template.algorithm_template import AlgorithmTemplate
from skmultiflow.drift_detection import DDM as DDM2

class DDM(AlgorithmTemplate):
    def __init__(self):
        self.ddm = DDM2(warning_level=2.2, out_control_level=2.4, min_num_instances=60)

    def add_data(self, data, info):
        self.ddm.add_element(data)
        # print("processed" + str(data))
        if self.ddm.detected_warning_zone():
            print('Warning zone has been detected in year: ' + info)
        if self.ddm.detected_change():
            print('Change has been detected in year: ' + info)

        return None
