#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import os
import numpy as np
from utils.ANN import ANN
from utils.Database import Database
import os
import matplotlib.pyplot as plt


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        #print(Di)
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)
    
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def get_features_corpus():
    ann = ANN()
    corpus = []
    labels = []
    shapes_ids = []
    db = Database()
    sql = f"""SELECT * FROM features"""
    db.cursor.execute(sql)
    rows = db.cursor.fetchall()
    for row in rows:
        shapes_ids.append(row[-2])
        label = get_label(shape_id = row[-2], db = db)
        labels.append(label)        
        features = ann.get_embedding_from_feature_row(row)
        corpus.append(np.array(features))
    db.close()
    return np.array(corpus), np.array(labels), np.array(shapes_ids)

def clean_corpus(corpus):
    corpus = corpus[~np.isnan(corpus).any(axis=1)]
    corpus = corpus[~np.isinf(corpus).any(axis=1)]
    return corpus

def normalize_corpus(corpus):
    # normalizing the corpus
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return corpus

def compute_tsne():
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    corpus, labels, shapes_ids = get_features_corpus()
    corpus = clean_corpus(corpus)
    corpus = normalize_corpus(corpus)
    
    with open(os.path.join("database", "indexes", "tsne_embedding_labels.npy"), "wb") as f:
        np.save(f, labels)

    with open(os.path.join("database", "indexes", "tsne_embedding_ids.npy"), "wb") as f:
        np.save(f, shapes_ids)
    
    Y = tsne(corpus, 2, corpus.shape[1], 20.0)
    
    with open(os.path.join("database", "indexes", "tsne_embedding.npy"), "wb") as f:
        np.save(f, Y)

    return Y, labels, shapes_ids

def get_label(shape_id, db):
    sql = f"""SELECT class FROM shapes WHERE id = {shape_id}"""
    db.cursor.execute(sql)
    label = db.cursor.fetchone()[0]
    return label

def load_tsne_and_labels():
    with open(os.path.join("database", "indexes", "tsne_embedding.npy"), "rb") as f:
        Y = np.load(f)
    with open(os.path.join("database", "indexes", "tsne_embedding_labels.npy"), "rb") as f:
        labels = np.load(f)
    
    with open(os.path.join("database", "indexes", "tsne_embedding_ids.npy"), "rb") as f:
        shapes_ids = np.load(f)
    
    label_colors = {}
    for label in set(labels):
        label_colors[label] = np.random.rand(3)
    
    return Y, labels, label_colors, shapes_ids

def plot_tsne():
    Y, labels, label_colors, shape_ids = load_tsne_and_labels()
    plt.figure(figsize=(10, 10))
    for i in range(len(Y)):
        plt.scatter(Y[i, 0], Y[i, 1], color=label_colors[labels[i]], label=labels[i])
    
    classes = np.unique(labels)
    plt.legend(classes, loc = 'upper right')
    ax = plt.gca()
    legend = ax.get_legend()
    for l in range(len(legend.legendHandles)):
        legend.legendHandles[l].set_color(label_colors[classes[l]])
    file_path = os.path.join("report", "tsne.png")
    plt.savefig(file_path)
    
if __name__ == "__main__":
    #compute_tsne() 
    plot_tsne()
