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

DATA_PATH = "data"
RESULTS_PATH = os.path.join("report", "evaluation_results", "results.npy")

def compute_confusion_matrix(k = 1):
    """
    Compute the confusion matrix for the ANN and the database
    :return: confusion matrix
    """
    console = Logger()
    
    classes = os.listdir(os.path.join(DATA_PATH, "PRINCETON", "train")) + os.listdir(os.path.join(DATA_PATH, "LabeledDB_new", "train"))
    confusion_matrix = np.zeros((len(classes), len(classes)))
    class_mapper = {classes[i]: i for i in range(len(classes))}
    
    query = QueryHandler(log=False)
    
    console.log("Computing confusion matrix...")
    for r,d,f in os.walk(DATA_PATH):
        if "train" in r:
            for file in tqdm(f):
                if file.endswith(".ply"):
                    class_true = r.split(os.sep)[-1]                    
                    path = os.path.join(r, file).replace("data", "preprocessed")
                    filenames, distances = query.get_similar_shapes_indexed(path, k)
                    for f_pred in filenames:
                        class_pred = f_pred.split(os.sep)[-2]                                       
                        confusion_matrix[class_mapper[class_true]][class_mapper[class_pred]] += 1
    console.success("Done!")
    
    console.log("Saving confusion matrix...")
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        np.save(f, confusion_matrix)
    console.success("Done!")
    return confusion_matrix

def load_confusion_matrix():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            return np.load(f)
    return None

def compute_accuracy(confusion_matrix):
    """
    Compute the accuracy of the ANN
    :param confusion_matrix: confusion matrix
    :return: accuracy
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def compute_precision(confusion_matrix):
    """
    Compute the precision of the ANN
    :param confusion_matrix: confusion matrix
    :return: precision
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)

def compute_recall(confusion_matrix):
    """
    Compute the recall of the ANN
    :param confusion_matrix: confusion matrix
    :return: recall
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

def compute_f1_score(confusion_matrix):
    """
    Compute the F1 score of the ANN
    :param confusion_matrix: confusion matrix
    :return: F1 score
    """
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    return 2 * precision * recall / (precision + recall)

def compute_evaluation_marks():
    confusion_matrix = load_confusion_matrix()
    acc = compute_accuracy(confusion_matrix)
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    f1_score = compute_f1_score(confusion_matrix)
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    
if __name__ == "__main__":
    compute_evaluation_marks()