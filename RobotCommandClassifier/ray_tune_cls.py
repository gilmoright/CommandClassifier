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

def load_data(data_config):
    df = pd.read_csv(data_config["path_to_df"])
    train_x_df = df.loc[df["subset"]=="train", data_config["input_column"]]
    train_y_df = df.loc[df["subset"]=="train", data_config["target_columns"]]
    test_x_df = df.loc[df["subset"]=="test", data_config["input_column"]]
    test_y_df = df.loc[df["subset"]=="test", data_config["target_columns"]]
    return train_x_df, train_y_df, test_x_df, test_y_df

def calculate_metrics(test_y_df, test_pred, config):
    metrics_for_report = {}
    if "correct_samples_perc" in config["report_metrics"]:
        totalAbsCorrect = 0
        for i in range(test_y_df.shape[0]):
            row = test_y_df.iloc[i]
            if (row.values==test_pred[i,:]).all():
                totalAbsCorrect += 1
        totalAbsCorrect = totalAbsCorrect / test_y_df.shape[0]
        metrics_for_report["correct_samples_perc"] = totalAbsCorrect
    if len(set(["avg_macro_f1", "avg_acc", "class_acc, class_macro_f1"]) & set(config["report_metrics"])) > 0:
        avg_macro_f1, avg_acc = 0, 0
        for i,c in enumerate(test_y_df.columns):
            cls_report = classification_report(test_y_df[c], test_pred[:,i], output_dict="True")
            avg_acc += cls_report["accuracy"]
            avg_macro_f1 += cls_report['macro avg']["f1-score"]
            if "class_acc" in config["report_metrics"]:
                metrics_for_report["[{}]_acc".format(c)] = cls_report["accuracy"]
            if "class_macro_f1" in config["report_metrics"]:
                metrics_for_report["[{}]_macrof1".format(c)] = cls_report['macro avg']["f1-score"]
        if "avg_macro_f1" in config["report_metrics"]:
            metrics_for_report["avg_macro_f1"] = avg_macro_f1 / len(test_y_df.columns)
        if "avg_acc" in config["report_metrics"]:
            metrics_for_report["avg_acc"] = avg_acc / len(test_y_df.columns)
    return metrics_for_report

def train_function(config):
    
    train_x_df, train_y_df, test_x_df, test_y_df = load_data(config["data_config"])
    pipeline_list = []
    if "TfidfVectorizer" in config["vectorizers"]:
        pipeline_list.append(("tfidf_vectorizer", TfidfVectorizer(**config["vectorizers"]["TfidfVectorizer"])))

    #classifier_class = globals()[config["classifier"]["class"]]
    if config["classifier"]["class"]=="RandomForestClassifier":
        classifier_class = RandomForestClassifier
    elif config["classifier"]["class"]=="ExtraTreesClassifier":
        classifier_class = ExtraTreesClassifier
    elif config["classifier"]["class"]=="ExtraTreeClassifier":
        classifier_class = ExtraTreeClassifier
    elif config["classifier"]["class"]=="DecisionTreeClassifier":
        classifier_class = DecisionTreeClassifier
    elif config["classifier"]["class"]=="KNeighborsClassifier":
        classifier_class = KNeighborsClassifier
    elif config["classifier"]["class"]=="RadiusNeighborsClassifier":
        classifier_class = RadiusNeighborsClassifier
    pipeline_list.append(("classifier", classifier_class(**config["classifier"]["params"])))

    pipe = Pipeline(pipeline_list)
    pipe.fit(train_x_df, train_y_df)
    test_pred = pipe.predict(test_x_df)
    
    metrics_for_report = calculate_metrics(test_y_df, test_pred, config)
    tune.report(**metrics_for_report)
    return metrics_for_report[config["target_metric"]]


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
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
