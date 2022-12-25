'''
Least-Squares Classifier
'''
import numpy as np
from sklearn.linear_model import LinearRegression

def Transform_X(X):
    X_t = []
    for item in X:
        item_t = np.append(item, [1])
        X_t.append(item_t)
    return X_t

def Transform_T(T, n):
    T_t = np.ones_like(T) * -1
    for i in range(len(T)):
        if T[i] == n:
            T_t[i] *= -1
    return T_t

def LSPredict(ts, wt, cn):
    ts_i = np.array(Transform_X(np.array(ts[0])))
    m = len(ts_i)
    prediction = np.ones(m)
    for i in range(m):
        pred_i = np.zeros(len(cn))
        for j in range(len(cn)):
            pred_i[j] = np.dot(ts_i[i], wt[j])
        prediction[i] = np.argmax(pred_i)
    correct = prediction[prediction == ts[1]]
    return len(correct) / len(ts[1]) * 100

def LSLearn(tr, cn):
    lr = 0.01
    max_iter = 10000
    tr_i = np.array(Transform_X(np.array(tr[0]))) # 50000 x 3072
    tr_l = np.array(tr[1])
    m = len(tr_i)
    d = len(tr_i[0])
    wt = [np.zeros(d) for _ in range(len(cn))]
    pi = np.linalg.pinv(tr_i)
    for j in range(len(cn)):
        wt[j] = np.dot(pi, Transform_T(tr_l, j))
    return wt