import keras
from keras.models import load_model
import cv2
import numpy as np
from libs.hdf5datasetgeneratormultilabel import HDF5DatasetGenerator
from libs.hdf5datasetwriter import HDF5DatasetWriter
model=load_model("ResNet50")
model1=keras.Sequential()
model.pop()
model.pop()
for layer in model.layers:
    model1.add(layer)
model=model1
print(model.summary())
BATCH_SIZE=32

trainGen = HDF5DatasetGenerator("hdf5/train2.hdf5", BATCH_SIZE, classes=5)
valGen = HDF5DatasetGenerator("hdf5/val.hdf5", BATCH_SIZE, classes=5)
print(trainGen.db["images"].shape[0])
print(valGen.db["images"].shape[0])
train_dataset=HDF5DatasetWriter((trainGen.db["images"].shape[0],2048),"hdf5/train_features_ResNet50.hdf5",dataKey="features",bufSize=1000)

features=model1.predict(trainGen.db["images"])
#f1=[]
#for feature in features:
#    f1.append(feature.flatten())
#features=np.array(f1)
print(features.shape)
train_dataset.add(features,trainGen.db["labels"])
train_dataset.close()



val_dataset=HDF5DatasetWriter((valGen.db["images"].shape[0],2048),"hdf5/val_features_ResNet50.hdf5",dataKey="features",bufSize=1000)

features=model1.predict(valGen.db["images"])
print(features.shape)
#f1=[]
#for feature in features:
#    f1.append(feature.flatten())
#features=np.array(f1)
val_dataset.add(features,valGen.db["labels"])
val_dataset.close()
trainGen.close()
valGen.close()
