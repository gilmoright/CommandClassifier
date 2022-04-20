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

def load_data(path_to_df, input_column="x", target_columns=[], test_only_on_fold=None, **kwargs):
    """
    Загрузка данных из csv файла path_to_df.
    input_column - колонка с текстом, которая будет использоваться как вход. Пока считаем, что она всегда одна
    target_columns - список колонок, которые содержат целевые классы
    test_only_on_fold - если тест разделён на фолды, и этот параметр не None, то выбираем только указанный фолд, остальное замешиваем в трейн.
    """
    df = pd.read_csv(path_to_df)
    if type(input_column)=="str":
        assert input_column in df.columns
    else:
        assert len(set(input_column) & set(df.columns))==len(input_column)
    assert len(set(target_columns) & set(df.columns))==len(target_columns)

    row_filter = df["subset"]=="train"
    if test_only_on_fold is not None:
        row_filter |= (df["fold"] != test_only_on_fold) & (df["subset"] == "test")
    train_x_df = df.loc[row_filter, input_column]
    train_y_df = df.loc[row_filter, target_columns]

    row_filter = df["subset"]=="valid"
    valid_x_df = df.loc[row_filter, input_column]
    valid_y_df = df.loc[row_filter, target_columns]

    row_filter = df["subset"]=="test"
    if test_only_on_fold is not None:
        row_filter &= df["fold"] == test_only_on_fold
    test_x_df = df.loc[row_filter, input_column]
    test_y_df = df.loc[row_filter, target_columns]
        
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