__author__ = 'fanxn'

import numpy as np
import matplotlib.pyplot as plt
import re


# Transform txt datafile with no column labels to available npy file
def data_trans(filename):
    tar_name = re.sub(r'\.\w+$', '.npy', filename)
    with open(filename, 'r') as ifs:
        result = []
        for line in ifs.readlines():
            result.append(line.strip().split(","))
        np.save(tar_name, result)


def computeCost(X, y, theta):
    m = len(y)
    cost_rslt = np.sum(np.dot(X, theta) - y) / (2 * m)
    return cost_rslt


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros((num_iters, 1), dtype=np.float64)

    for i in range(num_iters):
        XT = np.transpose(X)
        theta = theta - alpha / m * np.dot(XT, np.dot(X, theta) - y)
        cost_history[i] = computeCost(X, y, theta)

    return (theta, cost_history)


def featureNormalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_rslt = (X - mu) / sigma
    return (X_rslt, mu, sigma)

def normalEqn(X, y):
    XT = np.transpose(X)
    # 装换为方阵以便求逆
    X_reformed = np.dot(XT, X)
    theta_T = np.dot(np.linalg.inv(X_reformed), XT).dot(y)
    return theta_T

def proc():
    d1 = np.load('../../datas/ex1data1.npy')
    X = np.array(d1[:, 0], dtype=np.float64)
    y = np.array(d1[:, 1], dtype=np.float64)
    m = len(y)
    plt.figure(1)
    plt.plot(X, y)

    X = np.hstack((np.ones(m), X))
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    computeCost(X, y, theta)

    theta, _ = gradientDescent(X, y, theta, alpha, iterations)

    plt.legend(('Training data', 'Linear regression'))
    plt.plot(X[:, 1], np.dot(X, theta), "-")
    plt.savefig('../../static/pic/ex1data1.png')

    predict1 = np.dot(np.array([1, 3.5]), theta)
    predict2 = np.dot(np.array([1, 7]), theta)

    theta0_vals = np.linspace(-10, 100, 10)
    theta1_vals = np.linspace(-1, 100, 4)
    cost_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            cost_vals[i, j] = computeCost(X, y, t)

    cost_vals = np.transpose(cost_vals)

    d2 = np.load('../../datas/ex1data2.npy')
    X = np.ndarray((len(d2), 2), buffer=np.array(d2[:, 0:2], dtype=np.float64))
    # 若仍用一维数组,则加减运算会自动补全
    y = np.ndarray((len(d2), 1), buffer=np.array(d2[:, 2], dtype=np.float64))

    X, mu, sigma = featureNormalize(X)
    X = np.hstack((np.ones((len(d2), 1)), X))
    alpha = 0.01
    num_iters = 400
    theta1, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    # 查看矩阵形状用np.shape
    print(J_history.shape)

    theta = normalEqn(X, y)


if __name__ == "__main__":
    proc()
