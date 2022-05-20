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
        return 'Alright then, send the data to predict!'

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('predict_proba', type=str, location='args', default=False)
        args = parser.parse_args()

        json_data = request.get_json()
        df = pd.read_json(json_data)
        df.fillna(value=np.nan, inplace=True)

        if args.get('predict_proba', False):
            y_probas = loaded_model.predict_proba(df).tolist()
            return y_probas

        y_pred = loaded_model.predict(df).tolist()
        return y_pred


api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6789)
