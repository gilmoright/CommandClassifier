#!/usr/bin/env python
# coding: utf-8

import numpy as np
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

def binarymultilabel_to_multiclassmultilabel(raw_outputs, classes_per_attribute, CONFIG):
    """
    Функция переводит бинарный вектор, в котором каждая компонента - активность (от 0 до 1) какого-либо класса для какого-либо лейбла (атрибута)
    в вектор где каждая компоненты - целое число (номер класса) для одного лейбла.
    например [0,1,0,0,1,0,0,0,0,0,0,0,1] -> [1,0,4]
    params:
        raw_outputs - 2д массив (samples, number_of_binary_predictions)
        classes_per_attribute - количество классов внутри каждого лейбла.
        CONFIG - конфиг эксперимента
    return:
        2д массив (samples, number_of_labels)
    """
    predictions_2 = np.zeros((len(raw_outputs), len(classes_per_attribute)))
    for i in range(len(raw_outputs)):
        shift = 0
        for j in range(len(classes_per_attribute)):
            if "OneHotArgs" in CONFIG["Data"] and CONFIG["Data"]["OneHotArgs"].get('drop', None) == "first":
                # Эксперимент, когда для каждого класса предсказывается сигмойда, 
                # а если для лейбла нет класса с активностью превысевшей порог, то класс 0. 
                # для самого класса 0 при этом активности нет.
                label_logits = raw_outputs[i, shift:shift+classes_per_attribute[j]-1]
                # предполагается, что выходы модели были сигмойдами
                assert max(label_logits) < 1.0001
                if sum(label_logits>=0.5)==0:
                    predictions_2[i,j] = 0
                else:
                    predictions_2[i,j] = np.argmax(label_logits)+1
                shift += classes_per_attribute[j]-1
            else:
                predictions_2[i,j] = np.argmax(raw_outputs[i, shift:shift+classes_per_attribute[j]])
                shift += classes_per_attribute[j]
        assert shift==len(raw_outputs[0])
    return predictions_2

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