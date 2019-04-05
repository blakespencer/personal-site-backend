import os
from flask import Flask, send_from_directory, jsonify, request
import pickle
from spotipy.oauth2 import SpotifyClientCredentials
from secrets import spotify_information
import spotipy
import spotipy.util as util
import pandas as pd
import urllib


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


client_credentials_manager = SpotifyClientCredentials(
    client_id=spotify_information['client_id'], client_secret=spotify_information['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_spotify_row(uri):
    track = sp.track(uri)
    features = sp.audio_features(tracks=[uri])
    analysis = sp.audio_analysis(uri)
    track['album']['artists'][0]['track'] = track['name']
    track = pd.DataFrame(track['album']['artists'][0])[
        ['name', 'track']].reset_index()

    features = pd.DataFrame(features)

    analysis = analysis['track']
    analysis['codestring'] = None
    analysis['echoprintstring'] = None
    analysis['synchstring'] = None
    analysis['rhythmstring'] = None

    analysis = pd.DataFrame(analysis, index=[0])

    feat_ana = pd.merge(analysis, features)
    feat_ana = feat_ana._get_numeric_data()

    return pd.merge(feat_ana, track, how='left',  left_index=True, right_index=True).drop('index', axis=1)


@app.route('/api/search')  # the site to route to, index/main in this case
def search_spotify() -> str:
    try:
        prediction = jsonify(message='20000'), 200
        uri = request.args.get('query')
        if(uri is not None):
            query = urllib.parse.unquote(uri)
            response = sp.search(query, limit=12)
            tracks = response['tracks']['items']
            prediction = jsonify(tracks), 200
        return prediction
    except Exception as e:
        print(e)
        return jsonify([]), 404


with open('random_forest_100.pkl', 'rb') as picklefile:
    random_forest_model = pickle.load(picklefile)


@app.route('/api/data')  # the site to route to, index/main in this case
def get_predictions() -> str:
    prediction = jsonify(message='20000'), 200
    uri = request.args.get('uri')
    if(uri is not None):
        uri_edit = uri.split(':')[2]
        row = get_spotify_row(uri_edit)
        row = row[features]
        classes = random_forest_model.classes_
        probs = list(random_forest_model.predict_proba([row.iloc[0]])[0])
        probs_list = []
        track_info = list(row.iloc[0])
        for i in range(len(classes)):
            probs_list.append({'genre': classes[i], 'value': probs[i]})
        prediction = jsonify({'message': random_forest_model.predict([row.iloc[0]])[
            0], 'probs': probs_list, 'track_info': track_info}), 200

    return prediction


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists("build/" + path):
        return send_from_directory('build', path)
    else:
        return send_from_directory('build', 'index.html')


if __name__ == '__main__':
    app.run(use_reloader=True, port=5000, threaded=True)
