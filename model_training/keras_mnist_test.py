import numpy as np
import keras
import cv2

model = keras.models.load_model('model/keras_mnist.mod')
img = cv2.imread('sample_letters/bradhitc_40_O.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (28, 28))
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)
print('IMAGE', img.shape)

preds, = model.predict(img)
preds = list(preds)
for i in range(len(preds)):
    print('class', i, 'certainty', float(preds[i]))
