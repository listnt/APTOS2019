| WARNING: Все файлы PKL и файлы моделей пустышки. Файлы из папки libs временно неликвидны! |
| Многие скрипты были утеряны, эталонная папка - new_balance_t1t2 |
| --- |

## Обозначение приставок к названиям папок 
new\old - наличие\отсутсвие предобработки изображений
balance\without - наличие\отсутсвие балансировки данных
t1t2\OneT - наличие\отсутствие разбиения обучающего набора на 2 поднабора для SVM классификатора и CNN сети

1) необходимо скачать и распаковать датасет
2) переименовать папки в test и train
3) создать папки test_resize_224, train_resize_224, train_resize_224_balance, train_224_AAA, hdf5, output
4) запустить скрипты по порядку 

# описание скриптов

1.py<br/> 
данный скрипт считывает изображения из папки train и test
изменяет размер картинок и сохраняет в новую папку
**Утерян**

2.py<br/>
данный скрипт делает оверсаплинг классов изображений
**Утерян**

3.py<br/>
данный скрипт копирует картинки из результатов первого и второго скрипта в общую папку
**Утерян**

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


# Transfer learning. Модели которые не переобучались, а были взяты с весами "imagenet" из keras application

## ImageNet DenseNet 

SVM validation score: 0.427

Модель сохранена в pickle файл DenseNet50ImageNet.pkl

на kaggle даже заливать не буду, очень маленькая точность


## ImageNet ResNet 
SVM validation score: 0.545

Модель сохранена в pickle файл ResNet50ImageNet.pkl

на kaggle даже заливать не буду. очень маленькая точность
