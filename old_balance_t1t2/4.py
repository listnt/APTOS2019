#данный скрипт создает файлы h5py для загрузки в генератор обучения

from sklearn.model_selection import train_test_split
from libs.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import json
import cv2
import pandas as pd
from tqdm import tqdm
train=pd.read_csv("../../../train.csv")
train1=pd.read_csv("../newtrain.csv")
train=train.append(train1)

data_dir="../train_224_AAA/"

trainPaths=data_dir+train["id_code"]+".png"
trainLabels=train.diagnosis.values
test=pd.read_csv("../../../sample_submission.csv")
testPaths=list(paths.list_images("../../test_resize_224"))
testLabels=test.diagnosis.values
split = train_test_split(trainPaths, trainLabels,
test_size=500, stratify=trainLabels,
random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split
datasets = [
("train1", trainPaths[:4000], trainLabels[:4000], "hdf5/train1.hdf5"),
("train2", trainPaths[4000:], trainLabels[4000:], "hdf5/train2.hdf5"),
("val", valPaths, valLabels, "hdf5/val.hdf5"),
("test", testPaths, testLabels, "hdf5/test.hdf5")]
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 224, 224, 3), outputPath)
    for (i, (path, label)) in enumerate(tqdm(zip(paths, labels))):
        image = cv2.imread(path)
        if (dType == "train1") | (dType=="train2"):
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        if image is None:
            print(path)
        writer.add([image], [label])
    writer.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open("output/mean.json", "w")
f.write(json.dumps(D))
f.close()