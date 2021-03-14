import tensorflow as tf 
import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import pickle

model = ResNet50(weights='imagenet',include_top=False)

cap = cv2.VideoCapture('test.mp4')
frames = []
frame = None
ret = True
count=0
while(ret):
	ret, frame = cap.read()

	frame = np.resize(frame,(224,224,3))
	frame = np.expand_dims(frame,axis=0)
	frame = preprocess_input(frame)

	frames.append(frame)
	count+=1


	if not ret:
		break
	#cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()


frames = np.vstack(frames)

features = model.predict(frames)
features = features.reshape(features.shape[0],7*7*2048)
features = np.append(features,np.zeros((5,7*7*2048)),axis=0)
print(features.shape)

test_dataset = np.vsplit(features,10)

pickle.dump(test_dataset,open('test_dataset.pkl','wb'))


