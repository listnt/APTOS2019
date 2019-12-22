1) ���������� ������� � ����������� �������
2) ������������� ����� � test � train
3) ������� ����� test_resize_224, train_resize_224, train_resize_224_balance, train_224_AAA, hdf5, output
4) ��������� ������� �� ������� 
#�������� ��������
#1.py
������ ������ ��������� ����������� �� ����� train � test
�������� ������ �������� � ��������� � ����� �����
#2.py
������ ������ ������ ����������� ������� �����������
#3.py
������ ������ �������� �������� �� ����������� ������� � ������� ������� � ����� �����
#4.py
������ ������ ������� ����� h5py ��� �������� � ��������� �������� � ���������
#5.py
������ ������ ��������� ������� ��������
#6.py
������ ������ ���������� ���� � ������� ��������� �� ������� ���������� �� ������ ������� � �������([features],[labels])
#7.py
������ ������ ������� � ��������� svm ������
#6.imagenet.py
��� 6.py, ������ ����������� ������ � imagenet ������
#7.imagenet.py
������ ������ ������� � ��������� svm ������. ������������ �������� �� 6.imagenet.py �����



#������ ������� ������
8025 ����������� � ������������� ������ �������� �� 5 ������� �� 1500 ����������� � ������
1000 ����������� � ������������� �������� �� 5 �������. 200 ����������� � ������.



#�������� �������
������������� DenseNetModel (���� - DenseNetModel, ������� �������� �� �������� � output)
�������
def build_model(newmodel): #��� newmodel - Densenet121 �� keras application
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
�������� ~ 96 % �� ������������� � ������������.
�� kaggle �������� ��������� ~ 90 %(��������� 200 �������� �� train.csv). �������� �� ��������  0.778074
������ ��������� � ���� DenseNetModel

�������� ��������� SVM ����������� ��� ���� ������� - 0.901.
�� kaggle ���� �������


�������������  ResNetModel (���� - ResNetModel, ������� �������� �� �������� � output) 
�������
def build_model(newmodel): #��� newmodel - ResNet50 �� keras application
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
�������� ~ 98 % �� ������������� � ������������.
�� kaggle �������� ��������� ~ 90 %(��������� 200 �������� �� train.csv). �������� �� ��������  0.770888
������ ��������� � ���� ResnetModel


�������� ���������  SVM ����������� ��� ���� ������� - 0.91.
�� kaggle ���� �������

# Transfer learning. ������ ������� �� �������������, � ���� ����� � ������ "imagenet" �� keras application

ImageNet DenseNet 
SVM validation score: 0.427
�� kaggle ���� �������� �� ����, ����� ��������� ��������


ImageNet ResNet 
SVM validation score: 0.545
�� kaggle ���� �������� �� ����. ����� ��������� ��������