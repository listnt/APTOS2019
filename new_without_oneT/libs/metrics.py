from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import cohen_kappa_score
class Metrics(Callback):
    def my(self,valdata,name):
        self.validation_data=valdata
        self.name=name
    def on_train_begin(self, logs={}):
        self.val_kappas = []


    def on_epoch_end(self, epoch, logs={}):
        # print(self.validation_data[1])
        X_val, y_val = self.validation_data
        y_val = y_val.sum(axis=1) - 1

        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(self.name)

        return
