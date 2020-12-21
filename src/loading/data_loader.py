import random

import pandas as pd

columns_to_drop = ['Region', 'Country', 'State', 'City']
x = ['Month', 'Day']
y = ['AvgTemperature']

year = 1995

class DataLoader:
    def __init__(self):
        self.data = pd.read_csv("../data/city_temperature.csv")
        # print(self.data.columns)
        self.data = self.data[self.data['City'].isin(['Algiers'])]
        # self.data = self.data[self.data['Month'] == 1]
        self.data = self.data.sort_values(by='Year')
        self.data = self.data.drop(columns_to_drop, axis=1)
        # print(self.data)
        self.itr = -1
        self.indexes = self.data.index.values.tolist()
        self.outliers_idxs = []

    def apply_outliers(self, percentage, error):
        self.outliers_idxs = random.choices(self.indexes, k=int(percentage*len(self.indexes)))
        self.error = error


    def get_predict_data(self):
        return [self.data.loc[self.data['Year'].isin([year])][x].values, self.data.loc[self.data['Year'].isin([year])][y].values.ravel()]

    def restart_itr(self):
        self.itr = -1

    def get_data(self):
        if self.itr %100 == 0 and False:
            print(str(self.itr/len(self.indexes)))
        if self.itr > len(self.indexes)-2:
            return None
        else:
            self.itr += 1
            er = 0
            isOut = False
            if self.itr in self.outliers_idxs:
                er = self.error
                isOut = True
            return [self.data.iloc[[self.itr]][x].values[0], self.data.iloc[[self.itr]][y].values.ravel()[0] + er, isOut]
