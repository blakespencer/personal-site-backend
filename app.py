import os
from flask import Flask, send_from_directory, jsonify
import pickle

app = Flask(__name__, static_folder='react_app/build')

features = (['end_of_fade_in',
             'start_of_fade_out',
             'loudness',
             'tempo',
             'tempo_confidence',
             'time_signature',
             'time_signature_confidence',
             'acousticness',
             'danceability',
             'energy',
             'instrumentalness',
             'liveness',
             'speechiness',
             'valence'
             ])

# Serve React App

with open('prediction_object.pkl', 'rb') as picklefile:
    prediction_object = pickle.load(picklefile)

with open('histogram_data.pkl', 'rb') as picklefile:
    histogram_data = pickle.load(picklefile)

with open('precision_obj.pkl', 'rb') as picklefile:
    precision_data = pickle.load(picklefile)

with open('feature_importance.pkl', 'rb') as picklefile:
    feature_data = pickle.load(picklefile)


@app.route('/api/classification')
def get_classification_data() -> str:
    return jsonify({
        'featureImportance': feature_data,
        'histogram': histogram_data,
        'precision': precision_data,
        'wrongPrediction': prediction_object
    }), 200


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists("react_app/build/" + path):
        return send_from_directory('react_app/build', path)
    else:
        return send_from_directory('react_app/build', 'index.html')


if __name__ == '__main__':
    app.run(use_reloader=True, port=5000, threaded=True)
