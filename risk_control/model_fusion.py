import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from mlxtend.regressor import StackingRegressor
from mlxtend.classifier import StackingClassifier
from models import Models

model = Models()


# ---------------------------------------定义混合类----------------------------------------
class Blend:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train['y'].values
        self.y_test = y_test['y'].values

    def blending(self):
        mete_clf = LogisticRegression()
        clf1 = model.svm_classifier()
        clf2 = model.dt_classifier()
        # reg3 = model.xgb_classifier()
        self.blend = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=mete_clf)
        self.blend.fit(self.x_train, self.y_train)
        return self.blend

    def score(self):
        scores = cross_val_score(self.blend, X=self.x_train, y=self.y_train, cv=5,
                                 verbose=2)
        return scores

    def prediction(self):
        y_pred = self.blend.predict(self.x_test)
        return y_pred
