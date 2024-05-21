from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Function to create dummy data
def create_dummy_data():
    dates = pd.date_range(start='1/1/2020', periods=100)
    prices = np.random.rand(100) * 1000
    return pd.DataFrame({'Date': dates, 'Price': prices})

# Function to preprocess data
def preprocess_data(data):
    X = data[['Date']].apply(lambda x: x.factorize()[0]).values.reshape(-1, 1)  # Converting date to categorical integer
    y = data['Price'].values
    return X, y

@app.route('/dummy-data', methods=['GET'])
def get_dummy_data():
    data = create_dummy_data()
    return data.to_json(orient='records')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data_json = request.get_json()
    data = pd.DataFrame(data_json)
    X, y = preprocess_data(data)
    return jsonify({'X': X.tolist(), 'y': y.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
