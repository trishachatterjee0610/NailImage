import os
import io
#import requests
import flask
from flask import request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import PIL.Image as Image

CURRENT_DIRECTORY = os.curdir

# Initialize the flask application
app = Flask(__name__)

def load_baseline_model():

    checkpoint_filepath = f"{CURRENT_DIRECTORY}/model/baseline-cnn-model.hdf5"
    global baseline_model
    baseline_model = load_model(checkpoint_filepath)
    baseline_model._make_predict_function()


def load_vgg_model():

    checkpoint_filepath = f"{CURRENT_DIRECTORY}/model/vgg16-classifier-model.hdf5"
    global vgg_model
    vgg_model = load_model(checkpoint_filepath)
    vgg_model._make_predict_function()


def image_preparation(image, target):


    if image.mode != "GRAY":
        image = image.convert("GRAY")
    image = image.resize(target)
    image = np.array(img_to_array(image))

    image = np.expand_dims(image, axis=0)
    processed_image = preprocess_input(image)

    return processed_image


def interpreted_prediction(prediction):

    class_dict = {0: 'bad', 1: 'good'}
    rp = int(round(prediction[0][0]))
    return float(prediction[0][0]), rp, class_dict.get(rp)


@app.route("/predict", methods=["GET"])
def predict():


    data = {"success": False}

    image_url = request.args.get("image_url")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))


        image = image_preparation(image, target=(224, 224))


        prediction = vgg_model.predict(image)
        data = {"prediction": interpreted_prediction(prediction), "success": True}

    return flask.jsonify(data)

@app.route("/baseline/predict", methods=["GET"])

def baseline_predict():
    """
    Predict using baseline CNN model

    :return:
    """

    data = {"success": False}
    image_url = request.args.get("image_url")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))


        image = image_preparation(image, target=(224, 224))


        prediction = baseline_model.predict(image)
        data = {"prediction": interpreted_prediction(prediction), "success": True}

    return flask.jsonify(data)


if __name__ == "__main__":

    print((" *Starting server using FLASK_APP"))


    load_baseline_model()
    load_vgg_model()

    #app=socketio.Middleware(sio,app)
    #eventlet.wsgi.server(eventlet.listen(('', 4567,)),app)


    app.run()
