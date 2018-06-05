import numpy as np
import keras
import cv2

model = keras.models.load_model('model/keras_mnist.mod')
img = cv2.imread('base_letters/arial_numbers_2.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


print('MODEL', model.summary())
print('IMAGE', img.shape)
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)
print('IMAGE', img.shape)

preds = model.predict(img)
print(str(preds))
print(len(preds[0]))