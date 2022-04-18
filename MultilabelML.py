#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
assert torch.cuda.is_available()

import yaml
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import simpletransformers
from sklearn.metrics import classification_report
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs,
    ClassificationModel, ClassificationArgs
)
from RobotCommandClassifier import utils

if __name__=="__main__":
    import argparse
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
    with open(args.config_file, "r") as f:
        experiments = yaml.safe_load(f)
    EXPERIMENT_NAME = args.experiment_name
    CONFIG = experiments[EXPERIMENT_NAME]

    if os.path.exists(CONFIG["Report"]["report_dir"]) and len(os.listdir(CONFIG["Report"]["report_dir"]))>0:
        raise ValueError("report_dir is not empty: {}".format(CONFIG["Report"]["report_dir"]))
    if not os.path.exists(CONFIG["Report"]["report_dir"]):
        os.makedirs(CONFIG["Report"]["report_dir"])
    with open(os.path.join(CONFIG["Report"]["report_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f)

    # Prepare data
    train_x_df, train_y_df, valid_x_df, valid_y_df, test_x_df, test_y_df = utils.load_data(**CONFIG["Data"])

    if CONFIG["Type"] == "simple_ml_multilabel_classifier":
        enc = OneHotEncoder()
        enc.fit(train_y_df.values)

        labels = []
        encoded_labels = enc.transform(train_y_df.values).toarray().astype(int)
        for i in range(train_y_df.shape[0]):
            labels.append(encoded_labels[i].tolist())
        train_df = pd.DataFrame(list(zip(train_x_df, labels)))    
    else:
        train_df = pd.concat([train_x_df, train_y_df], axis=1)
    train_df.columns = ["text", "labels"]

    # Create model
    if CONFIG["Type"] == "simple_ml_multilabel_classifier":
        model_args = MultiLabelClassificationArgs(**CONFIG["Model"]["Args"])
        # Create a MultiLabelClassificationModel
        model = MultiLabelClassificationModel(
            CONFIG["Model"]["model_type"],
            CONFIG["Model"]["model_name"],
            num_labels=len(labels[0]),
            use_cuda=True,    
            args=model_args,
        )
    elif CONFIG["Type"] == "simple_ml_classifier":
        model_args = ClassificationArgs(**CONFIG["Model"]["Args"])
        model = ClassificationModel(
            CONFIG["Model"]["model_type"],
            CONFIG["Model"]["model_name"],
            num_labels=len(train_y_df.iloc[:,0].unique()),
            use_cuda=True,    
            args=model_args,
        )
    else:
        raise ValueError("unknown Type of experiment:{}".format(CONFIG["Type"]))

    # Train the model
    model.train_model(train_df, overwrite_output_dir=True)

    # Predict and evaluate
    predictions, raw_outputs = model.predict(valid_x_df.values.tolist())
    if CONFIG["type"] == "simple_ml_multilabel_classifier":
        predictions_2 = np.zeros((len(predictions), len(enc.categories_)))
        for i in range(len(predictions)):
            shift = 0
            for j in range(len(enc.categories_)):
                predictions_2[i,j] = np.argmax(raw_outputs[i, shift:shift+len(enc.categories_[j])])
                shift += len(enc.categories_[j])
        predictions = predictions_2
    result = utils.calculate_metrics_2(valid_y_df, predictions, display=True)
    with open(os.path.join(CONFIG["Report"]["report_dir"], "classes_report.json"), "w") as f:
        json.dump(result, f)

    result_avg = utils.calculate_metrics(valid_y_df, predictions, config={
        "report_metrics": CONFIG["Report"]["report_metrics"]
    })
    with open(os.path.join(CONFIG["Report"]["report_dir"], "avg_report.json"), "w") as f:
        json.dump(result_avg, f)
    for k, v in result_avg.items():
        print(np.round(v*100), "\t", k)