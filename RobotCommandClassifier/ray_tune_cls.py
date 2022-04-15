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
import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import time

from RobotCommandClassifier import utils


def train_function(config):
    
    train_x_df, train_y_df, valid_x_df, valid_y_df, _, _ = utils.load_data(config["data_config"])
    pipeline_list = []
    if "TfidfVectorizer" in config["vectorizers"]:
        pipeline_list.append(("tfidf_vectorizer", TfidfVectorizer(**config["vectorizers"]["TfidfVectorizer"])))

    classifier_class = utils.CLASSIFIERS_NAME_TO_CLASS[config["classifier"]["class"]]
    pipeline_list.append(("classifier", classifier_class(**config["classifier"]["params"])))

    pipe = Pipeline(pipeline_list)
    pipe.fit(train_x_df, train_y_df)
    valid_pred = pipe.predict(valid_x_df)
    
    metrics_for_report = utils.calculate_metrics(valid_y_df, valid_pred, config)
    tune.report(**metrics_for_report)
    return metrics_for_report[config["target_metric"]]

subspaces_for_classifiers = {
    "RandomForestClassifier": {
            "class": "RandomForestClassifier",
            "params": {
                "n_estimators": tune.randint(10,500),
                "criterion": tune.choice(["gini", "entropy"]),
                "min_samples_split": tune.randint(2, 20),
                "min_samples_leaf": tune.randint(2, 10),
                "max_features": tune.choice(["auto", "sqrt", "log2"]),
                "bootstrap": True, #tune.choice([True, False]),
                #"class_weight": hp.choice([None, "balanced", "balanced_subsample"]),
                "max_samples": tune.uniform(0.,1.0)

            }
        },
    "ExtraTreesClassifier": {
        "class": "ExtraTreesClassifier",
        "params": {
                "n_estimators": tune.randint(10,500),
                "criterion": tune.choice(["gini", "entropy"]),
                "min_samples_split": tune.randint( 2, 20),
                "min_samples_leaf": tune.randint( 2, 10),
                "max_features": tune.choice(["auto", "sqrt", "log2"]),
                "bootstrap": True, #tune.choice([True, False]),
                #"class_weight": hp.choice([None, "balanced", "balanced_subsample"]),
                "max_samples": tune.uniform(0.,1.0)
        },
    },
    "ExtraTreeClassifier": {
            "class": "ExtraTreeClassifier",
            "params": {
                "criterion": tune.choice(["gini", "entropy"]),
                "splitter": tune.choice(["best", "random"]),
                "min_samples_split": tune.randint(2, 20),
                "min_samples_leaf": tune.randint( 2, 10),
                "max_features": tune.choice(["auto", "sqrt", "log2"]),
            },
        },
    "DecisionTreeClassifier": {
            "class": "DecisionTreeClassifier",
            "params": {
                "criterion": tune.choice( ["gini", "entropy"]),
                "splitter": tune.choice(["best", "random"]),
                "min_samples_split": tune.randint( 2, 20),
                "min_samples_leaf": tune.randint(2, 10),
                "max_features": tune.choice(["auto", "sqrt", "log2"]),
            },
        },
    "KNeighborsClassifier": {
            "class": "KNeighborsClassifier",
            "params": {
                "n_neighbors" : tune.randint( 3,20),
                "weights": tune.choice(["uniform", "distance"]),
                "algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size" : tune.randint(10,50),
                "p" : tune.randint(1,3)
            },
        },
    "RadiusNeighborsClassifier": {
            "class": "RadiusNeighborsClassifier",
            "params": {
                "radius": tune.randn(1, 1),
                "n_neighbors" : tune.randint(3,20),
                "weights": tune.choice(["uniform", "distance"]),
                "algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size" : tune.randint( 10,50),
                "p" : tune.randint(1,3)
            },
        }
}
config_space = {
    "data_config" : {
        "path_to_df": "/s/ls4/users/grartem/RL_robots/CommandClassifier/Data/Interim/split.csv",
        "target_columns": ['y', 'action', 'direction', 'meters', 'degshours',
                           'object1', 'nearest', 'relation1', 'object2', 
                           'relation2', 'object3'],
        "input_column": "x"
    },
    "vectorizers": {
        "TfidfVectorizer" : {
            "lowercase": True,
            "analyzer": tune.choice(["word", "char", "char_wb"]),
            "ngram_range": tune.choice([(1,1), (1,2),(1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,4), (5,5)]),
            "max_df": tune.uniform(0.7,1.0),
            "min_df": tune.randint(1, 100),
            "max_features": tune.randint(1000, 10000),
            "norm": tune.choice(["l1", "l2"]),
            "use_idf": tune.choice([True, False]),
            "smooth_idf": tune.choice([True, False]),
            "sublinear_tf": tune.choice([True, False])
        }
    },
    "report_metrics": ["correct_samples_perc", "class_acc", "class_macro_f1", "avg_acc", "avg_macro_f1"],
    # target_metric  должна также присутствовать в report_metrics
    "target_metric": "avg_macro_f1"
}


def run_hyperopt_tune(config_space):
    print("START TIME:", time.asctime())
    algo = HyperOptSearch(space=config_space, metric="avg_macro_f1", mode="max")
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    analysis = tune.run(
        train_function,
        metric="avg_macro_f1",
        mode="max",
        name=config_space["classifier"]["class"],
        search_alg=algo,
        scheduler=scheduler,
        num_samples=500,
        local_dir="/s/ls4/users/grartem/RL_robots/CommandClassifier/Results"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    print("END TIME:", time.asctime())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune some sklearn algorythms")
    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--algo_class",
        default=None,
        type=str,
        help="FullName of Algorythm class from sklearn")
    args = parser.parse_args()
    algo_subspace = subspaces_for_classifiers[args.algo_class]
    config_space["classifier"] = algo_subspace
    ray.init(configure_logging=False)
    run_hyperopt_tune(config_space)
