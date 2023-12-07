import scipy.io as sio
import time
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings("ignore")


def experiment(A_, G, X, gnd, k):
    # Set the order of filter
    G_ = G.copy()
    kk = 1

    result = [0, 0, 0, 0, 0]
    # acc_list = []
    # nmi_list = []
    # f1_list = []
    # nowa = []
    # nowk = []
    # best_acc = []
    # best_nmi = []
    # best_f1 = []
    # best_a = []
    # best_k = []

    # Set the list of alpha
    list_a = [1e-4, 1e-2, 1, 10, 100]

    # Set the range of filter order k
    while kk <= 5:

        # compute
        X_bar = G_.dot(X)  # eq(3.3)
        XtX_bar = X_bar.dot(X_bar.T)  # XXt_bar in fact
        XXt_bar = X_bar.T.dot(X_bar)  # XtX_bar in fact
        # tmp_acc = []
        # tmp_nmi = []
        # tmp_f1 = []
        # tmp_a = []
        for a in list_a:
            tmp = np.linalg.inv(I2 + XXt_bar / a)
            tmp = X_bar.dot(tmp).dot((X_bar.T))
            tmp = I / a - tmp / (a * a)
            S = tmp.dot(a * A_ + XtX_bar)
            C = 0.5 * (np.fabs(S) + np.fabs(S.T))
            # print("a={}".format(a), "k={}".format(kk))
            u, s, v = sp.linalg.svds(C, k=k, which='LM')

            kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
            predict_labels = kmeans.predict(u)

            cm = clustering_metrics(gnd, predict_labels)
            ac, nm, f1 = cm.evaluationClusterModelFromLabel()

            if ac > result[0]:
                result = [ac, nm, f1, a, kk]
        #     acc_list.append(ac)
        #     nmi_list.append(nm)
        #     f1_list.append(f1)
        #     nowa.append(a)
        #     nowk.append(kk)
        #
        #     tmp_acc.append(ac)
        #     tmp_nmi.append(nm)
        #     tmp_f1.append(f1)
        #     tmp_a.append(a)
        #         a = a + 50
        # nxia = np.argmax(tmp_acc)
        # best_acc.append(tmp_acc[nxia])
        # best_nmi.append(tmp_nmi[nxia])
        # best_f1.append(tmp_f1[nxia])
        # best_a.append(tmp_a[nxia])
        # best_k.append(kk)
        kk += 1
        G_ = G_.dot(G)

    return result


def paper(A, G, X, gnd, k):
    A_ = A.copy()
    Poly_A = A.copy()
    result = [0, 0, 0, 0, 0, ""]
    for i in range(1, 10):
        Poly_A = Poly_A.dot(A)
        A_ = A_ + Poly_A
        func = "A"
        for j in range(i):
            func += " + A^{}".format(j + 2)

        ex = experiment(A_, G, X, gnd, k)

        if ex[0] > result[0]:
            result = ex
            result.append(func)
    return result


def ours(A, G, X, gnd, k):
    A_ = A.copy()
    result = [0, 0, 0, 0, 0, ""]
    for i in range(1, 10):
        A_ += A
        func = "A*{}".format(i + 1)

        ex = experiment(A_, G, X, gnd, k)
        if ex[0] > result[0]:
            result = ex
            result.append(func)
    return result


if __name__ == '__main__':
    dataset_list = ['cora', 'citeseer', 'wiki', 'ACM3025', 'large_cora', 'pubmed']
    for dataset in dataset_list:
        data = sio.loadmat('../doc/{}.mat'.format(dataset))
        if dataset == 'large_cora':
            X = data['X']
            A = data['G']
            gnd = data['labels']
            gnd = gnd[0, :]
        elif dataset == 'ACM3025':
            X = data['feature']
            A = data['PAP']
            gnd = data['label']
            gnd = gnd.T
            gnd = gnd - 1
            gnd = gnd[0, :]
        else:
            X = data['fea']
            A = data['W']
            gnd = data['gnd']
            gnd = gnd.T
            gnd = gnd - 1
            gnd = gnd[0, :]

        # Store some variables
        N = X.shape[0]
        k = len(np.unique(gnd))
        I = np.eye(N)
        I2 = np.eye(X.shape[1])
        if sp.issparse(X):
            X = X.todense()

        # Normalize A
        A = A + I
        D = np.sum(A, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        A = D.dot(A).dot(D)

        # Get filter G
        Ls = I - A  # normalized graph Laplacian
        G = I - 0.5 * Ls

        print("dataset: {}".format(dataset))
        with open('../doc/result.txt', 'a') as f:
            f.write("{}:\n".format(dataset))

        result = paper(A, G, X, gnd, k)
        ex = ours(A, G, X, gnd, k)
        if ex[0] > result[0]:
            result = ex
        print("f(A) = {}, a = {}, k = {}\nacc = {}, nmi = {}, f1 = {}\n".format(
                result[5], result[3], result[4], result[0], result[1], result[2]))
        with open('../doc/result.txt', 'a') as f:
            f.write("f(A) = {}, a = {}, k = {}\nacc = {}, nmi = {}, f1 = {}\n\n".format(
                result[5], result[3], result[4], result[0], result[1], result[2]))

