# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os

from utils.QueryHandler import QueryHandler
from utils.Logger import Logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd   

DATA_PATH = "data"
RESULTS_PATH = os.path.join("report", "evaluation_results")

def compute_confusion_matrix(k = 1, method = "ANN"):
    """
    Compute the confusion matrix for the ANN and the database
    :return: confusion matrix
    """
    console = Logger()
    
    classes = os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train")) + os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train"))
    confusion_matrix = np.zeros((len(classes), len(classes)))
    class_mapper = {classes[i]: i for i in range(len(classes))}
    
    query = QueryHandler(log=False)
    
    methods = {
        "ANN": query.get_similar_shapes_indexed,
        "KNN": query.find_similar_shapes
    }
    
    console.log("Computing confusion matrix...")
    for r,d,f in os.walk(DATA_PATH):
        if "train" in r:
            for file in tqdm(f):
                if file.endswith(".ply"):
                    class_true = r.split(os.sep)[-1]                    
                    path = os.path.join(r, file).replace("data", "preprocessed")
                    filenames, distances = methods[method](filename = path, k = k)
                    for f_pred in filenames:
                        class_pred = f_pred.split(os.sep)[-2]                                    
                        confusion_matrix[class_mapper[class_true]][class_mapper[class_pred]] += 1
    console.success("Done!")
    console.log("Saving confusion matrix...")
    folder = os.path.join(RESULTS_PATH, method)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"confusion_matrix_k_{k}.npy"), "wb") as f:
        np.save(f, confusion_matrix)
    console.success("Done!")
    return confusion_matrix

def load_confusion_matrix(k=1, method="ANN"):
    path = os.path.join(RESULTS_PATH, method, f"confusion_matrix_k_{k}.npy")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return np.load(f)
    return None

def plot_confusion_matrix(confusion_matrix, k=1, method="ANN"): 
    classes = os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train")) + os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train"))
    confusion_matrix = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    # Normalizing confusion matrix...
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
    #confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    plt.figure(figsize=(30, 20))
    sns.heatmap(confusion_matrix, annot=True, cmap="Greens", fmt=".2f")
    folder = os.path.join(RESULTS_PATH, method)
    plt.savefig(os.path.join(folder, f"confusion_matrix_k_{k}.png"))
    plt.show()

def compute_accuracy(confusion_matrix):
    """
    Compute the accuracy
    :param confusion_matrix: confusion matrix
    :return: accuracy
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def compute_roc_curve(confusion_matrix):
    """_summary_
        Compute the ROC curve
    Args:
        confusion_matrix (_type_): _description_
    """
    true_negatives = np.sum(confusion_matrix) - np.sum(np.diag(confusion_matrix))
    false_positives = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    specificity = true_negatives / (true_negatives + false_positives)
    false_positive_rate = 1 - specificity
    
    return specificity, false_positive_rate

def compute_precision(confusion_matrix):
    """
    Compute the precision 
    :param confusion_matrix: confusion matrix
    :return: precision
    """
    true_pos = np.diag(confusion_matrix)
    false_pos = np.sum(confusion_matrix, axis=0) - true_pos
    return np.sum(true_pos / (true_pos + false_pos))

def compute_recall(confusion_matrix):
    """
    Compute the recall 
    :param confusion_matrix: confusion matrix
    :return: recall
    """
    true_pos = np.diag(confusion_matrix)
    false_neg = np.sum(confusion_matrix, axis=1) - true_pos
    return np.sum(true_pos / (true_pos + false_neg))

def compute_f1_score(confusion_matrix):
    """
    Compute the F1 score 
    :param confusion_matrix: confusion matrix
    :return: F1 score
    """
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    return 2 * precision * recall / (precision + recall)

def compute_evaluation_marks(k=1, method="ANN"):
    confusion_matrix = load_confusion_matrix(k=k, method=method)
    # Normalizing confusion matrix...
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    
    # Computing evaluation marks...
    acc = compute_accuracy(confusion_matrix)
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    f1_score = compute_f1_score(confusion_matrix)
    specificity, false_positive_rate = compute_roc_curve(confusion_matrix)
    
    console = Logger()
    console.success(f"Accuracy: {acc}")
    console.success(f"Precision: {precision}")
    console.success(f"Recall: {recall}")
    console.success(f"F1 score: {f1_score}")
    
    # saving the metrics...
    with open(os.path.join(RESULTS_PATH, method, f"metrics_k_{k}.txt"), "w") as f:
        f.write("Accuracy, Precision, Recall, F1 score\n")
        f.write(f"{acc}, {precision}, {recall}, {f1_score}")
        f.close()
    
    plt.title("ROC curve for k = " + str(k) + " and method = " + method)
    plt.plot(specificity, false_positive_rate)
    plt.savefig(os.path.join(RESULTS_PATH, method, f"roc_curve_k_{k}.png"))
    plt.show()

def plot_roc_curve_for_k(k=1):
    cm1 = load_confusion_matrix(k=k, method="ANN")
    cm2 = load_confusion_matrix(k=k, method="KNN")
    sp1, fpr1 = compute_roc_curve(cm1)
    sp2, fpr2 = compute_roc_curve(cm2)
    plt.title("ROC curve for k = " + str(k))
    plt.plot(sp1, fpr1, label="ANN")
    plt.plot(sp2, fpr2, label="KNN")
    plt.savefig(os.path.join(RESULTS_PATH, f"roc_curve_k_{k}.png"))
    plt.show()

def plot_class_histogram_prediction(class_name = "Human", k_max = 3):
    classes = os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train")) + os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train"))
    preds1 = []
    preds2 = []
    for k in range(1, k_max + 1):
        cm1 = pd.DataFrame(load_confusion_matrix(k=k, method="ANN"), index=classes, columns=classes)
        cm2 = pd.DataFrame(load_confusion_matrix(k=k, method="KNN"), index=classes, columns=classes)
        preds1.append(cm1.loc[class_name, class_name] / k)
        preds2.append(cm2.loc[class_name, class_name] / k)
    
    path = os.path.join(RESULTS_PATH, "histograms_per_class")
    os.path.makedirs(path, exist_ok=True)

    class_instances = len(os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train", class_name))) + len(os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train", class_name)))
    
    plt.title(f"Predictions for class {class_name} with method ANN")
    plt.ylim(0, class_instances)
    plt.hist(preds1, label="ANN")
    plt.savefig(os.path.join(path, f"histogram_{class_name}_ANN.png"))

    plt.cla()
    plt.title(f"Predictions for class {class_name} with method KNN")
    plt.ylim(0, class_instances)
    plt.hist(preds2, label="KNN")
    plt.savefig(os.path.join(path, f"histogram_{class_name}_KNN.png"))
        
if __name__ == "__main__":
    k = 1
    method = "KNN"
    
    compute_confusion_matrix(k, method)
    compute_evaluation_marks(k, method)
    
    confusion_matrix = load_confusion_matrix(k=k, method=method)
    plot_confusion_matrix(confusion_matrix, k, method)
    