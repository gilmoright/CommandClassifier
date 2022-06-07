#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
assert torch.cuda.is_available()

import logging
import yaml
import json
import pyhocon
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import simpletransformers
from sklearn.metrics import classification_report
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs,
    ClassificationModel, ClassificationArgs
)
from RobotCommandClassifier.MyMultilabel import MyMultiLabelClassificationModel, MyMultiLabelClassificationArgs
from RobotCommandClassifier import utils

def ValidateConfig_setupExperiment():
    parser = argparse.ArgumentParser(description="Tune some sklearn algorythms")
    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="path to config file")
    parser.add_argument(
        "--experiment_name",
        default=None,
        type=str,
        help="name of experiment to get from config file")
    args = parser.parse_args()
    # setup the experiment
    if ".conf" in args.config_file:
        configFileContent = pyhocon.ConfigFactory.parse_file(args.config_file)
        CONFIG = configFileContent[args.experiment_name].as_plain_ordered_dict()
    else:
        with open(args.config_file, "r") as f:
            experiments = yaml.safe_load(f)    
        CONFIG = experiments[args.experiment_name]
    if os.path.exists(CONFIG["output_dir"]) and len(os.listdir(CONFIG["output_dir"]))>0:
        raise ValueError("report_dir is not empty: {}".format(CONFIG["output_dir"]))
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])
    with open(os.path.join(CONFIG["output_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f)

    return CONFIG

def PrepareModel(CONFIG):
    # Create model
    CONFIG["Model"]["Args"]["output_dir"] = CONFIG["output_dir"]+"/models/"
    CONFIG["Model"]["Args"]["best_model_dir"] = CONFIG["output_dir"] + "/models/best_model"
    if CONFIG["Type"] == "simple_ml_multilabel_classifier":
        model_args = MultiLabelClassificationArgs(**CONFIG["Model"]["Args"])
        model = MultiLabelClassificationModel(
            CONFIG["Model"]["model_type"],
            CONFIG["Model"]["model_name"],
            CONFIG["Model"]["num_labels"],
            use_cuda=True,    
            args=model_args,
        )
    elif CONFIG["Type"] == "simple_ml_classifier":
        model_args = ClassificationArgs(**CONFIG["Model"]["Args"])
        model = ClassificationModel(
            CONFIG["Model"]["model_type"],
            CONFIG["Model"]["model_name"],
            CONFIG["Model"]["num_labels"],
            use_cuda=True,    
            args=model_args,
        )
    elif CONFIG["Type"] == "mymulti_classifier":
        model_args = MyMultiLabelClassificationArgs(**CONFIG["Model"]["Args"])
        # Create a MultiLabelClassificationModel
        model = MyMultiLabelClassificationModel(
            CONFIG["Model"]["model_type"],
            CONFIG["Model"]["model_name"],
            CONFIG["Model"]["num_labels"],
            use_cuda=True,
            num_sublabels_per_biglabel = CONFIG["Model"]["num_sublabels_per_biglabel"],
            add_attention_for_labels=CONFIG["Model"]["add_attention_for_labels"],
            args=model_args,
        )
    else:
        raise ValueError("unknown Type of experiment:{}".format(CONFIG["Type"]))
    return model

def PrepareData(CONFIG):
    # Prepare data
    train_x_df, train_y_df, valid_x_df, valid_y_df, test_x_df, test_y_df = utils.load_data(**CONFIG["Data"])
    
    # Добавляем описание y к входному тексту, чтобы указать на тип шаблона
    if CONFIG["Data"].get("add_y_to_x", False):
        with open(CONFIG["Data"]["y_descriptions_path"], "r") as f:
            y_descriptioons = json.load(f)
        train_x_df = train_x_df["y"].map(lambda y: y_descriptioons[int(y)]) + ": " + train_x_df["x"]
        valid_x_df = valid_x_df["y"].map(lambda y: y_descriptioons[int(y)]) + ": " + valid_x_df["x"]
        test_x_df = test_x_df["y"].map(lambda y: y_descriptioons[int(y)]) + ": " + test_x_df["x"]
    
    # если предсказываем только активность лейблов, то всё, что не 0 - делаем равным единице
    if CONFIG["Data"].get("predict_label_flag", False):
        if "y" in CONFIG["Data"]["target_columns"]:
            raise ValueError("Указан флаг для использования только бинарных лейблов. Предполагается, что в таком случае 'y' не должен ыть среди target_columns.")
        train_y_df[train_y_df!=0] = 1
        valid_y_df[valid_y_df!=0] = 1
        test_y_df[test_y_df!=0] = 1
    return train_x_df, train_y_df, valid_x_df, valid_y_df, test_x_df, test_y_df

        
def PrepareInput_for_simpletransformers(train_x_df, train_y_df, CONFIG):
    # Готовим инпут для сетки
    # Приводим вектор из целых чисел в  вектор из нулей и единиц
    # TODO: убрать num_labels в конфиг
    enc = None
    if CONFIG["Type"] in ["simple_ml_multilabel_classifier", "mymulti_classifier"]:
        # если предсказываем только активность, то это не надо делать на самом деле
        if CONFIG["Data"].get("predict_label_flag", False):
            labels = train_y_df.values.tolist()
        else:
            enc = OneHotEncoder(**CONFIG["Data"].get("OneHotArgs", {}))
            enc.fit(train_y_df.values)
    
            labels = []
            encoded_labels = enc.transform(train_y_df.values).toarray().astype(int)
            for i in range(train_y_df.shape[0]):
                labels.append(encoded_labels[i].tolist())
            print("labels", labels)
        train_df = pd.DataFrame(list(zip(train_x_df, labels)))        
    else:
        # Скорее всего это просто одноклассовый классификатор, можно не приводить к one-hot
        train_df = pd.concat([train_x_df, train_y_df], axis=1)
    train_df.columns = ["text", "labels"]

    if CONFIG["Type"] == "simple_ml_classifier":
        num_labels = len(train_y_df.iloc[:,0].unique())
    else:
        num_labels = len(labels[0])
        print("num_labels", num_labels)
    return train_df, enc, num_labels

if __name__=="__main__":
    import argparse
    
    CONFIG = ValidateConfig_setupExperiment()
    
    train_x_df, train_y_df, valid_x_df, valid_y_df, test_x_df, test_y_df = PrepareData(CONFIG)
    train_df, enc, num_labels = PrepareInput_for_simpletransformers(train_x_df, train_y_df, CONFIG)
    if "num_labels" in CONFIG["Model"]:
        assert CONFIG["Model"]["num_labels"] == num_labels
    else:
        CONFIG["Model"]["num_labels"] = num_labels
        
    joblib.dump(enc, CONFIG["output_dir"]+"/y_onehotenc.joblib")
    model = PrepareModel(CONFIG)
    # Train the model
    model.train_model(train_df, overwrite_output_dir=True)
