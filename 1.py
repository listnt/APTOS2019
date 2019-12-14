# данный скрипт считывает изображения из папки train и test
# изменяет размер картинок и сохраняет в новую папку
from shutil import copyfile
import os
import pandas as pd
from imutils import paths
from PIL import Image
from tqdm import tqdm


train=pd.read_csv("train.csv")
for some in tqdm(train.id_code.values):
    name=some+".png"
    img=Image.open(os.path.join("train",name))
    img=img.resize((224, )*2, resample=Image.LANCZOS)
    img.save(os.path.join("train_resize_224",name))


test=pd.read_csv("test.csv")
for some in tqdm(test.id_code.values):
    name=some+".png"
    img=Image.open(os.path.join("test",name))
    img=img.resize((224, )*2, resample=Image.LANCZOS)
    img.save(os.path.join("test_resize_224",name))
