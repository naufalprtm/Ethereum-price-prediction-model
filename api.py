import pandas as pd
import numpy as np

def create_dummy_data():
    dates = pd.date_range(start='1/1/2020', periods=100)
    prices = np.random.rand(100) * 1000
    return pd.DataFrame({'Date': dates, 'Price': prices})

def preprocess_data(data):
    X = data[['Date']].apply(lambda x: x.factorize()[0]).values.reshape(-1, 1)  # Converting date to categorical integer
    y = data['Price'].values
    return X, y
