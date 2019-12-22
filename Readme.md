1) необходимо скачать и распаковать датасет
2) переименовать папки в test и train
3) создать папки test_resize_224, train_resize_224, train_resize_224_balance, train_224_AAA, hdf5, output
4) запустить скрипты по порядку 

# описание скриптов

1.py<br/>
данный скрипт считывает изображения из папки train и test
изменяет размер картинок и сохраняет в новую папку


2.py<br/>
данный скрипт делает оверсаплинг классов изображений


3.py<br/>
данный скрипт копирует картинки из результатов первого и второго скрипта в общую папку


4.py<br/>
данный скрипт создает файлы h5py для загрузки в генератор обучения и валидации

5.py<br/>
данный скрипт запускает процесс обучения

6.py<br/>
данный скрипт генерирует файл с картами признаков из моделей полученных из пятого скрипта и метками([features],[labels])

7.py<br/>
данный скрипт обучает и проверяет svm машину

6.imagenet.py<br/>
как 6.py, только загружаются модели с imagenet весами

7.imagenet.py<br/>
данный скрипт обучает и проверяет svm машину. используются признаки из 6.imagenet.py файла


# Размер наборов данных

8025 изображений в тренировочном наборе разбитых по 5 классам по 1500 изображений в каждом<br/>
1000 изображений в валидационном разбитых по 5 классам. 200 изображений в каждом.



# Описание моделей

## DenseNet121

Переобученная DenseNetModel (файл - DenseNetModel, процесс обучения на картинке в output)
функция

```python
def build_model(newmodel): #где newmodel - Densenet121 из keras application
    model = Sequential()
    model.add(newmodel)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    return model
```
точность ~ 96 % на тренировочном и валидационом.

На kaggle точность валидации ~ 90 %(случайные 200 картинок из train.csv). Точность на тестовом  0.778074

Модель сохранена в файл DenseNetModel


## svm DenseNet121

точность валидации SVM построенная над этой моделью - 0.901.

Модель сохранена в pickle файл DenseNet121.pkl

на kaggle пока считает

## ResNet121

Переобученная  ResNetModel (файл - ResNetModel, процесс обучения на картинке в output) 
функция

```python
def build_model(newmodel): #где newmodel - ResNet50 из keras application
    model = Sequential()
    model.add(newmodel)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )

    return model
```

точность ~ 98 % на тренировочном и валидационом.

На kaggle точность валидации ~ 90 %(случайные 200 картинок из train.csv). Точность на тестовом  0.770888

Модель сохранена в файл ResnetModel

## svm ResNet50

точность валидации  SVM построенная над этой моделью - 0.91.

Модель сохранена в pickle файл ResNet50.pkl

на kaggle пока считает

# Transfer learning. Модели которые не переобучались, а были взяты с весами "imagenet" из keras application

## ImageNet DenseNet 

SVM validation score: 0.427

Модель сохранена в pickle файл DenseNet50ImageNet.pkl

на kaggle даже заливать не буду, очень маленькая точность


## ImageNet ResNet 
SVM validation score: 0.545

Модель сохранена в pickle файл ResNet50ImageNet.pkl

на kaggle даже заливать не буду. очень маленькая точность
