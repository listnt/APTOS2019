# данный скрипт запускает процесс обучения
from libs.hdf5datasetgeneratormultilabel import HDF5DatasetGenerator
from keras import layers
from keras.applications import DenseNet121, MobileNetV2
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
from libs.trainingmonitor import TrainingMonitor
from libs.metrics import Metrics
import os


def build_model(model):
    model = Sequential()
    model.add(model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )

    return model



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()



BATCH_SIZE = 32

trainGen = HDF5DatasetGenerator("hdf5/train.hdf5", BATCH_SIZE, classes=5)
valGen = HDF5DatasetGenerator("hdf5/val.hdf5", BATCH_SIZE, classes=5)



mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))




model = build_model(mobilenet)
labels = np_utils.to_categorical(valGen.db["labels"][:-1],5)
multilabels = np.empty(labels.shape, dtype=labels.dtype)
multilabels[:, 4] = labels[:, 4]
for i in range(3, -1, -1):
    multilabels[:, i] = np.logical_or(labels[:, i], multilabels[:, i + 1])
labels=multilabels
kappa_metrics = Metrics()
kappa_metrics.my(valdata=(valGen.db["images"][:-1],labels))
path = os.path.sep.join(["output", "{}.png".format(
os.getpid())])
mycallback=TrainingMonitor(path)

history = model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // BATCH_SIZE,
    epochs=20,
    max_queue_size=BATCH_SIZE,
    callbacks=[kappa_metrics,mycallback], verbose=0)

print("[INFO] serializing model...")
model.save("output/model.model", overwrite=True)
trainGen.close()
valGen.close()