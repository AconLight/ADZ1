import pandas as pd

columns_to_drop = ['Region', 'Country', 'State', 'City']
x = ['Month', 'Day', 'AvgTemperature']
y = ['City']

year = 1995

class DataLoader:
    def __init__(self):
        self.data = pd.read_csv("../data/city_temperature.csv")
        # print(self.data.columns)
        self.data = self.data[self.data['City'].isin(['Algiers', 'Columbus'])]
        self.data = self.data[self.data['Month'] == 1]
        self.data = self.data.sort_values(by='Year')
        # self.data = self.data.drop(columns_to_drop, axis=1)
        # print(self.data)
        self.itr = -1
        self.indexes = self.data.index.values.tolist()

    def get_predict_data(self):
        return [self.data.loc[self.data['Year'].isin([year])][x].values, self.data.loc[self.data['Year'].isin([year])][y].values.ravel()]

    def get_data(self):
        if self.itr %100 == 0 and False:
            print(str(self.itr/len(self.indexes)))
        if self.itr > len(self.indexes)-2:
            return None
        else:
            self.itr += 1
            return [self.data.iloc[[self.itr]][x].values[0], self.data.iloc[[self.itr]][y].values.ravel()[0], self.data.iloc[[self.itr]]['Year'].values.ravel()[0]]
