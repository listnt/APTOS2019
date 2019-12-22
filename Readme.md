1) необходимо скачать и распаковать датасет
2) переименовать папки в test и train
3) создать папки test_resize_224, train_resize_224, train_resize_224_balance, train_224_AAA, hdf5, output
4) запустить скрипты по порядку 

# описание скриптов

1.py
данный скрипт считывает изображения из папки train и test
изменяет размер картинок и сохраняет в новую папку

2.py

данный скрипт делает оверсаплинг классов изображений

3.py

данный скрипт копирует картинки из результатов первого и второго скрипта в общую папку

4.py

данный скрипт создает файлы h5py для загрузки в генератор обучения и валидации

5.py

данный скрипт запускает процесс обучения

6.py

данный скрипт генерирует файл с картами признаков из моделей полученных из пятого скрипта и метками([features],[labels])

7.py

данный скрипт обучает и проверяет svm машину

6.imagenet.py

как 6.py, только загружаются модели с imagenet весами

7.imagenet.py

данный скрипт обучает и проверяет svm машину. используются признаки из 6.imagenet.py файла




# Размер наборов данных

8025 изображений в тренировочном наборе разбитых по 5 классам по 1500 изображений в каждом
1000 изображений в валидационном разбитых по 5 классам. 200 изображений в каждом.



# Описание моделей

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

точность валидации SVM построенная над этой моделью - 0.901.
на kaggle пока считает


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


точность валидации  SVM построенная над этой моделью - 0.91.
на kaggle пока считает

# Transfer learning. Модели которые не переобучались, а были взяты с весами "imagenet" из keras application

ImageNet DenseNet 
SVM validation score: 0.427
на kaggle даже заливать не буду, очень маленькая точность


ImageNet ResNet 
SVM validation score: 0.545
на kaggle даже заливать не буду. очень маленькая точность
