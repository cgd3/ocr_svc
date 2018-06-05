import json
import flask
import keras
import pickle

app_uri = '/ocr'
app_port = 6000
app = flask.Flask(__name__)
with open('model_training/model/keras_mnist.mod', 'rb') as infile:
    model = pickle.load(infile)


def inference(img):
    pred = model.predict(img)


@app.route(app_uri, methods=['POST'])
def default():
    try:
        request = flask.request
    except Exception as exc:
        return json.dumps({'service': exc})


if __name__ == '__main__':
    app.run(port=app_port)
