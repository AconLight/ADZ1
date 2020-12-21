import sesd

class OutTemplate:
    def __init__(self):
        pass

    def check(self, data):
        pass


class SHESD(OutTemplate):
    def __init__(self, max_anomalies):
        self.max_anomalies = max_anomalies

    def check(self, data):
        result = []
        data_to_check = []
        for i in data:
            result.append([False, i[2]])
            data_to_check.append(i[1])

        outs = sesd.seasonal_esd(data_to_check, hybrid=True, max_anomalies=self.max_anomalies)
        for o in outs:
            result[o][0] = True

        return result

