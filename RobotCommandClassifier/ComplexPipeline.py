#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, multilabel_confusion_matrix
import time

def extract_units_from_line(x):
    match = re.findall("(\d+ ?.*)|(один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать|час|полчаса) ? .*", x)
    if match is None:
        return ""
    return '+'.join(["".join(x) for x in match])

class Classifier_v1():
    """
    Классификатор, который отдельно классифицирует классы из списка
    """
    def __init__(self, columns, main_labels, branch_labels, branch_y_classes, fit_branch_on_gold=True):
        self.columns = columns
        self.main_labels = main_labels
        self.branch_labels = branch_labels
        self.branch_y_classes = branch_y_classes
        self.fit_branch_on_gold = fit_branch_on_gold
        self.main_pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=5000)),
            ("classifier", RandomForestClassifier(bootstrap=False)),
        ])
        self.branch_classifier = Pipeline([
            ("extracter", FunctionTransformer(lambda series: series.map(extract_units_from_line))),
            ("vectorizer", TfidfVectorizer(max_features=1000, analyzer="char", ngram_range=(2,3))),
            ("classifier", RandomForestClassifier()),
        ])

    def fit(self, X, Y):
        self.main_pipeline.fit(X, Y.drop(self.branch_labels, axis=1))
        if self.fit_branch_on_gold:
            X_branch = X[Y["y"].isin(self.branch_y_classes)]
            Y_branch = Y.loc[Y["y"].isin(self.branch_y_classes), self.branch_labels]
            self.branch_classifier.fit(X_branch, Y_branch)
        else:
            raise NotImplementedError
    
    def predict(self, X):
        final_pred = np.zeros((X.shape[0], len(self.columns)))
        main_pred = self.main_pipeline.predict(X)
        for i, c in enumerate(self.main_labels):
            final_pred[:,self.columns.index(c)] = main_pred[:,i]
        X_branch = X[np.isin(main_pred[:,0], self.branch_y_classes)]
        if len(X_branch) > 0:
            branch_pred = self.branch_classifier.predict(X_branch)
            for i, c in enumerate(self.branch_labels):
                final_pred[np.isin(main_pred[:,0], self.branch_y_classes),self.columns.index(c)] = branch_pred[:,i]
        return final_pred
