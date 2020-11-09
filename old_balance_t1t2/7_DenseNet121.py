from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import argparse
import pickle
import h5py
train_db = h5py.File("hdf5/train_features_DenseNet121.hdf5", "r")
val_db=h5py.File("hdf5/val_features_DenseNet121.hdf5", "r")
i = int(train_db["labels"].shape[0])
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0],"gamma":["scale","auto"]}
print("start")
#model = GridSearchCV(SVC(), params, cv=3,n_jobs=-1,verbose=10)
model=SVC(C=0.01,gamma="auto")
model.fit(train_db["features"][:i], train_db["labels"][:i])
#print("[INFO] best hyperparameters: {}".format(model.best_params_))


print("[INFO] evaluating...")
preds = model.predict(val_db["features"])
print(preds)
acc = accuracy_score(val_db["labels"], preds)
print("[INFO] score: {}".format(acc))

import pickle
filename = 'DenseNet121.pkl'
pickle.dump(model,open(filename,"wb"))