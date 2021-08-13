import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_openml 
import seaborn as sns 
import matplotlib.pyplot as plt 
import cv2  
from PIL import Image 
import PIL.ImageOps
import os, ssl, time


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context
X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)
#print(nclasses)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2500, train_size = 7500)
X_train_scaled = X_train/255
X_test_scaled = X_test/255
model = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled, y_train)
y_predict = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

cap = cv2.VideoCapture(0)

while True:
    try:
        Dummy,frame = cap.read()
        #print(Dummy)
        #cv2.rectangle(frame,(10,10),(100,100), (255,0,0), 1)
        #cv2.rectangle(frame, (200,200), (0,0), (120, 102, 120), 5)
        grey= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(grey.shape)
        height, width = grey.shape
        upper_left = int(width/2 -56), int(height/2-56)
        lower_left = int(width/2 +56), int(height/2+56)
        cv2.rectangle(grey, upper_left, lower_left,(0,255,0), 3)

        #region of interest (area)

        roi = grey[upper_left[1]:lower_left[1], upper_left[0]:lower_left[0]]
        #print(roi)

        image_pil = Image.fromarray(roi)
        #print(image_pil)
        image_between = image_pil.convert("L")

        im_resized = image_between.resize((28,28), Image.ANTIALIAS)
        #print(im_resized)

        im_resized_inverted = PIL.ImageOps.invert(im_resized)
        #print(im_resized)

        pix_filter = 25
        min_pix = np.percentile(im_resized_inverted,pix_filter)
        im_resized_inverted_scaled = np.clip(im_resized_inverted-min_pix, 0, 255)

        max_pix = np.max(im_resized_inverted)
        im_resized_inverted_scaled = np.asarray(im_resized_inverted_scaled)/max_pix
        #print(im_resized_inverted_scaled)
        test_sample = np.array(im_resized_inverted_scaled-min_pix).reshape(1,784)
        test_predict = model.predict(test_sample)
        print("predict_classes", test_predict)

        cv2.imshow("picture",grey)
        if cv2.waitKey(5) == ord("q"):
            break

    
    except Exception as e:
        pass

cv2.release()
cv2.destroyAllWindows()