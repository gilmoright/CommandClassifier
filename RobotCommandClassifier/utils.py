#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, multilabel_confusion_matrix

CLASSIFIERS_NAME_TO_CLASS = {
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreeClassifier": ExtraTreeClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier
}

def load_data(data_config):
    df = pd.read_csv(data_config["path_to_df"])
    train_x_df = df.loc[df["subset"]=="train", data_config["input_column"]]
    train_y_df = df.loc[df["subset"]=="train", data_config["target_columns"]]
    valid_x_df = df.loc[df["subset"]=="valid", data_config["input_column"]]
    valid_y_df = df.loc[df["subset"]=="valid", data_config["target_columns"]]
    test_x_df = df.loc[df["subset"]=="test", data_config["input_column"]]
    test_y_df = df.loc[df["subset"]=="test", data_config["target_columns"]]
        
    return train_x_df, train_y_df, valid_x_df, valid_y_df, test_x_df, test_y_df

def calculate_metrics(y_df, predict, config):
    metrics_for_report = {}
    if "correct_samples_perc" in config["report_metrics"]:
        totalAbsCorrect = 0
        for i in range(y_df.shape[0]):
            row = y_df.iloc[i]
            if (row.values==predict[i,:]).all():
                totalAbsCorrect += 1
        totalAbsCorrect = totalAbsCorrect / y_df.shape[0]
        metrics_for_report["correct_samples_perc"] = totalAbsCorrect
    if len(set(["avg_macro_f1", "avg_acc", "class_acc, class_macro_f1"]) & set(config["report_metrics"])) > 0:
        avg_macro_f1, avg_acc = 0, 0
        for i,c in enumerate(y_df.columns):
            cls_report = classification_report(y_df[c], predict[:,i], output_dict="True")
            avg_acc += cls_report["accuracy"]
            avg_macro_f1 += cls_report['macro avg']["f1-score"]
            if "class_acc" in config["report_metrics"]:
                metrics_for_report["[{}]_acc".format(c)] = cls_report["accuracy"]
            if "class_macro_f1" in config["report_metrics"]:
                metrics_for_report["[{}]_macrof1".format(c)] = cls_report['macro avg']["f1-score"]
        if "avg_macro_f1" in config["report_metrics"]:
            metrics_for_report["avg_macro_f1"] = avg_macro_f1 / len(y_df.columns)
        if "avg_acc" in config["report_metrics"]:
            metrics_for_report["avg_acc"] = avg_acc / len(y_df.columns)
    return metrics_for_report

def calculate_metrics_2(y_df, predict, display=True):
    if display:
        for i,c in enumerate(y_df.columns):
            print(c)
            print(classification_report(y_df[c], predict[:,i]))
    metrics = {}
    avg = 0
    for i,c in enumerate(y_df.columns):
        res = classification_report(y_df[c], predict[:,i], output_dict=True)
        metrics[c] = res
        avg += res['macro avg']["f1-score"]
    avg /= len(y_df.columns)
    metrics["avg"] = {
        "macro-f1": avg
    }
    return metrics