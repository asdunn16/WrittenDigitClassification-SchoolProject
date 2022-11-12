import cv2
import tensorflow as tf
import numpy as np
import imutils
from imutils.contours import sort_contours

# load model
model = tf.keras.models.load_model('digits_model.h5')
model.load_weights('digits_model_weights')

# test from training data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

predictions = model.predict([x_test])
print('label:', y_test[10])
print('prediction:', np.argmax(predictions[10]))

# load test image, convert to grayscale, blur it to reduce noise
image = cv2.imread('test2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection
edged = cv2.Canny(blurred, 30, 150)
contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method='left-to-right')[0]
# initialize chars list
chars = []

# loop over the contours
for c in contours:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # filter out bounding boxes
    if (5 <= w <= 150) and (12 <= h <= 120):
        # extract the character and apply Otsu’s Binarization
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        # if the width is greater than the height, resize along the width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=28)
        # else resize along the height
        else:
            thresh = imutils.resize(thresh, height=28)

        # pad to get 28 x 28
        (tH, tW) = thresh.shape
        dX = int(max(0, 28 - tW) / 2.0)
        dY = int(max(0, 28 - tH) / 2.0)

        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (28, 28))

        # prepare image for classification
        padded = padded.astype('float32') / 255.0
        padded = np.expand_dims(padded, axis=-1)

        # update our list of characters
        chars.append((padded, (x, y, w, h)))


chars = np.array([c[0] for c in chars], dtype='float32')

# make predictions
preds = model.predict(chars)

# define the list of label names
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# initialize output
output = []

for pred in preds:
    i = np.argmax(pred)
    prob = pred[i]
    label = label_names[i]
    output.append(label)

# show original image and print output
print('The output is:', ''.join(output))
cv2.imshow('Input Image', image)
cv2.waitKey(0)
