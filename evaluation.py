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
from sklearn.metrics import roc_curve
from utils.statistics import plot_bars   

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
    y_true, y_pred = [], []
    console.log("Computing confusion matrix...")
    for r,d,f in os.walk(DATA_PATH):
        if "train" in r:
            for file in tqdm(f):
                if file.endswith(".ply"):
                    class_true = r.split(os.sep)[-1]                    
                    path = os.path.join(r, file).replace("data", "preprocessed")
                    filenames, _ = methods[method](filename = path, k = k)
                    for f_pred in filenames:
                        class_pred = f_pred.split(os.sep)[-2]                                    
                        confusion_matrix[class_mapper[class_true]][class_mapper[class_pred]] += 1
                        y_true.append(class_true)
                        y_pred.append(class_pred)
                        
    console.success("Done!")
    console.log("Saving confusion matrix...")
    folder = os.path.join(RESULTS_PATH, method)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"confusion_matrix_k_{k}.npy"), "wb") as f:
        np.save(f, confusion_matrix)
    with open(os.path.join(folder, f"y_true_k_{k}.npy"), "wb") as f:
        np.save(f, y_true)
    with open(os.path.join(folder, f"y_pred_k_{k}.npy"), "wb") as f:
        np.save(f, y_pred)    
    console.success("Done!")
    return confusion_matrix, y_true, y_pred

def load_confusion_matrix(k=1, method="ANN"):
    path = os.path.join(RESULTS_PATH, method, f"confusion_matrix_k_{k}.npy")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return np.load(f)
    return None

def load_y(k=1, method="ANN"):
    path_y_true = os.path.join(RESULTS_PATH, method, f"y_true_k_{k}.npy")
    path_y_pred = os.path.join(RESULTS_PATH, method, f"y_pred_k_{k}.npy")
    
    if os.path.exists(path_y_pred) and os.path.exists(path_y_true):
        with open(path_y_true, "rb") as f:
            y_true = np.load(f)
        with open(path_y_pred, "rb") as f:
            y_pred = np.load(f)
        return y_true, y_pred
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
    plt.close()
    
def compute_accuracy(confusion_matrix):
    """
    Compute the accuracy
    :param confusion_matrix: confusion matrix
    :return: accuracy
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def compute_roc_curve(y_true, y_pred, cls):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = (y_true == cls).astype(int)
    y_pred = (y_pred == cls).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, cls

def compute_roc_curves(y_true, y_pred):
    """_summary_
        Compute the ROC curve
    Args:
        confusion_matrix (_type_): _description_
    """
    return [ compute_roc_curve(y_true, y_pred, cls) for cls in np.unique(y_true) ]
    
def compute_precision(confusion_matrix):
    """
    Compute the precision 
    :param confusion_matrix: confusion matrix
    :return: precision
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)

def compute_recall(confusion_matrix):
    """
    Compute the recall 
    :param confusion_matrix: confusion matrix
    :return: recall
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

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
    precision = list(compute_precision(confusion_matrix))
    recall = list(compute_recall(confusion_matrix))
    f1_score = list(compute_f1_score(confusion_matrix))
    
    y_true, y_pred = load_y(k=k, method=method)
    RCM = compute_roc_curves(y_true, y_pred)

    console = Logger()
    console.success(f"Accuracy: {acc}")
    console.log("")
    console.success(f"Precision: {precision}")
    console.log("")
    console.success(f"Recall: {recall}")
    console.log("")
    console.success(f"F1 score: {f1_score}")
    
    # saving the metrics...
    with open(os.path.join(RESULTS_PATH, method, f"metrics_k_{k}.txt"), "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 score: {f1_score}\n")
                
        f.close()
    
    plt.figure(figsize=(10, 10))
    plt.title("ROC curve for k = " + str(k) + " and method = " + method)
    for fpr, tpr, c in RCM:
        plt.plot(fpr, tpr, label=c)
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("Sensitivity")
    plt.savefig(os.path.join(RESULTS_PATH, method, f"roc_curve_k_{k}.png"))    
    plt.close()

def plot_roc_curve_for_k(k=1, cls = "Human"):
    y_true1, y_pred1 = load_y(k=k, method="ANN")
    y_true2, y_pred2 = load_y(k=k, method="KNN")
    fpr1, sen1, _ = compute_roc_curve(y_true1, y_pred1, cls)
    fpr2, sen2, _ = compute_roc_curve(y_true2, y_pred2, cls)
    plt.figure(figsize=(10, 10))
    plt.title("ROC curve for k = " + str(k) + " and class = " + cls)
    plt.plot(fpr1, sen1, label="ANN")
    plt.plot(fpr2, sen2, label="KNN")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("Sensitivity")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, f"roc_curve_k_{k}.png"))
    plt.close()

def plot_class_histogram_prediction(class_name = "Human", k_max = 1):
    classes = os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train")) + os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train"))
    preds1 = []
    preds2 = []
    for k in range(1, k_max + 1):
        cm1 = pd.DataFrame(load_confusion_matrix(k=k, method="ANN"), index=classes, columns=classes)
        cm2 = pd.DataFrame(load_confusion_matrix(k=k, method="KNN"), index=classes, columns=classes)
        preds1.append(cm1.loc[class_name, class_name] / k)
        preds2.append(cm2.loc[class_name, class_name] / k)
    
    path = os.path.join(RESULTS_PATH, "histograms_per_class")
    os.makedirs(path, exist_ok=True)

    if os.path.exists(os.path.join(DATA_PATH, "PRINCETON", "train", class_name)):
        class_instances = len(os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train", class_name))) 
    else:
        class_instances = len(os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train", class_name)))
    path = os.path.join(RESULTS_PATH, "histograms_per_class", class_name)
    os.makedirs(path, exist_ok=True)
    bins = [str(i) for i in range(1, k_max +1)]
    plot_bars(data = preds1, bins = bins, y_lim = class_instances, title=f"Predictions for class {class_name} with ANN", filename = os.path.join(path, f"histogram_{class_name}_ANN.png"))
    plot_bars(data = preds2, bins = bins, y_lim = class_instances, title=f"Predictions for class {class_name} with KNN", filename = os.path.join(path, f"histogram_{class_name}_KNN.png"))
    plt.close()
    
def compute_evaluation():
    method = "KNN"
    #compute_confusion_matrix(k, method)
    compute_evaluation_marks(k, method)
   
        
if __name__ == "__main__":
    #compute_evaluation()
    for k in range(1, 6):
        plot_roc_curve_for_k(k = k, cls = "Human")
    