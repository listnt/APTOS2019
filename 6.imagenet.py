import keras
from keras.models import load_model
import cv2
import numpy as np
from libs.hdf5datasetgeneratormultilabel import HDF5DatasetGenerator
from libs.hdf5datasetwriter import HDF5DatasetWriter
from keras.models import Sequential
from keras.applications import ResNet50,DenseNet121
BATCH_SIZE=32

densenet=DenseNet121(weights="imagenet",include_top=False,input_shape=(224,224,3))

model = Sequential()
model.add(densenet)
model.add(keras.layers.GlobalAveragePooling2D())

print(model.summary)
trainGen = HDF5DatasetGenerator("hdf5/train.hdf5", BATCH_SIZE, classes=5)
valGen = HDF5DatasetGenerator("hdf5/val.hdf5", BATCH_SIZE, classes=5)
print(trainGen.db["images"].shape[0])
print(valGen.db["images"].shape[0])
train_dataset=HDF5DatasetWriter((trainGen.db["images"].shape[0],1024),"hdf5/train_features_DenseNetImageNet.hdf5",dataKey="features",bufSize=1000)

features=model.predict(trainGen.db["images"])
print(features.shape)
train_dataset.add(features,trainGen.db["labels"])
train_dataset.close()



val_dataset=HDF5DatasetWriter((valGen.db["images"].shape[0],1024),"hdf5/val_features_DenseNetImageNet.hdf5",dataKey="features",bufSize=1000)

features=model.predict(valGen.db["images"])
print(features.shape)
val_dataset.add(features,valGen.db["labels"])
val_dataset.close()
trainGen.close()
valGen.close()
