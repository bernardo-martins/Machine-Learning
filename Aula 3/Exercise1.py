import numpy as np
from funcoes import poly_16features, create_plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def calc_fold(feats, X, Y, train_ix, valid_ix, C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    prob = reg.predict_proba(X[:, :feats])[:, 1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])

def load_expand_standard_startifysplitkfold_data(file_name):
    data = np.loadtxt(file_name, delimiter=",")
    features = data[:, 1:]
    features_expanded = poly_16features(features)
    means = np.mean(features_expanded, axis=0)
    stdevs = np.std(features_expanded, axis=0)

    features_standard = (features_expanded - means) / stdevs

    final_data = np.insert(features_standard, 0, data[:, 0], axis=1)

    np.random.shuffle(final_data)
    final_data_shuffled = final_data
    Xs = final_data_shuffled[:, 1:]
    Ys = final_data_shuffled[:, 0]

    return Xs, Ys


def logisticreg_crossvalidation(Xs, Ys):
    X_r, X_t, Y_r, Y_t = train_test_split(Xs, Ys, test_size=0.33,
                                          stratify=Ys)
    errs = []
    folds = 5
    kf = StratifiedKFold(n_splits=5)
    for feats in range(2, 17):
        tr_err = va_err = 0
        for (tr_ix, va_ix) in kf.split(Y_r, Y_r):
            r, v = calc_fold(feats, X_r, Y_r, tr_ix, va_ix)
            tr_err += r
            va_err += v
            # print(feats, ':', tr_err/folds, va_err/folds)
        errs.append((tr_err/folds, va_err/folds))

    errs = np.array(errs)
    plt.figure(figsize=(8, 8), frameon=False)
    plt.plot(range(2, 17), errs[:, 0], "-b", linewidth=3)
    plt.plot(range(2, 17), errs[:, 1], "-r", linewidth=3)
    plt.show()


Xs, Ys = load_expand_standard_startifysplitkfold_data("data.txt")
logisticreg_crossvalidation(Xs, Ys)
