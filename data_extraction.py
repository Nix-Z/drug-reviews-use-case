import pandas as pd

def load_data():
    df_druglib_train = pd.read_csv('drugLibTest_raw.csv')
    df_druglib_test = pd.read_csv('drugLibTrain_raw.csv')
    data = pd.concat([df_druglib_train,df_druglib_test])
    print(data.head())
    return data

# load_data()
