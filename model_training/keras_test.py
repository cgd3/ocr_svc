import numpy as np
#import keras
import cv2

#model = keras.models.load_model('model/keras_mnist.mod')
img = cv2.imread('base_letters/calibri_lower_m.png')
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


#print('MODEL', model.summary())
print('IMAGE', img.shape)
test = np.asarray([img])
print('TEST', test.shape)

for sample in test:
    for x in sample:
        for y in x:
            y = np.asarray(y[0])

print('TEST', test.shape)

#preds = model.predict(test)
#print(preds)