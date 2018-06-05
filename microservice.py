import json
import flask
import keras
import pickle

app_port = 6000
app = flask.Flask(__name__)
with open('model_training/model/keras_mnist.mod', 'rb') as infile:
    model = pickle.load(infile)


def inference(img):
    pred = model.predict(img)


@app.route('/mnist', methods=['POST'])
def default():
    try:
        request = flask.request
        # request should just be string representing a 28x28 ndarray
        # like '[[255, 255], [255, 255]]'
    except Exception as exc:
        return json.dumps({'service': exc})


if __name__ == '__main__':
    app.run(port=app_port)
