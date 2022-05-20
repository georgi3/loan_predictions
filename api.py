from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from supporting_funcs import *
import pandas as pd
import pickle


app = Flask(__name__)
api = Api(app)
model_fname = './data/bernNB.pkl'


with open(model_fname, 'rb') as f:
    loaded_model = pickle.load(f)


class Predict(Resource):
    def get(self):
        return 'Alright then'

    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        y_probas = loaded_model.predict_proba(df).tolist()
        return y_probas


api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6789)
