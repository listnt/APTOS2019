# данный скрипт делает оверсаплинг классов изображений

from imutils import paths
import numpy as np
import collections
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import random
import os

aug = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
train = pd.read_csv("train.csv")
trainLabels = train.diagnosis.values
trainPaths = list(train.id_code.values)

mydict = dict(collections.Counter(trainLabels))
maxelt = max(mydict.values())
print(maxelt)
newtrain = pd.DataFrame(columns=["id_code", "diagnosis"])
for key in mydict.keys():
    indexes = []
    for i, el in enumerate(trainLabels):
        if el == key:
            indexes.append(i)
    indexes = np.array(indexes)
    elt = mydict[key]
    print(elt,maxelt)
    while elt < maxelt:
        elt = elt + 1
        augind = random.choice(indexes)
        imgname = trainPaths[augind]
        newimage = cv2.imread("./train_resize_224/" + imgname + ".png")
        newimage = aug.random_transform(newimage, seed=random.randint(0, 100000000))

        cv2.imwrite(os.path.join("./train_resize_224_balance/", imgname+str(elt)+".png"), newimage)
        newtrain = newtrain.append({"id_code": imgname+str(elt), "diagnosis": trainLabels[augind]},
                                   ignore_index=True)


newtrain.to_csv("newtrain.csv", index=False)
