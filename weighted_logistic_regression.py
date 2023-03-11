"""
Weighted_Logistic Regression
"""

import plotly.graph_objects as go
import numpy as np


def read_txt_as_float(path, separator=" "):
    ls = []
    with open(path, 'r') as file:
        for line in file:
            data = filter(lambda x: x != '', line.split(separator))
            ls.append(list(map(lambda x: float(x), data)))
    return np.array(ls)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def phi(data_train, option=2):
    p = []
    p.append(lambda x: np.transpose([[1, data[0], data[1] ** 2] for data in x]))
    p.append(lambda x: np.transpose([[1, data[0], data[1], data[0] * data[1], data[0] ** 2 * data[1]]
                                     for data in x]))
    p.append(lambda x: np.transpose([[1, data[0], data[1], data[0] * data[1], data[0] ** 2 * data[1],
                                      data[0] * data[1] ** 2] for data in x]))
    return p[option](data_train)


def train(train_data, data_point, label, phi_x, landa, tau):
    magnitude = lambda vector: sum(map(lambda x: x ** 2, vector))
    W = [np.exp((-0.5) * magnitude(data_point - x) / (tau ** 2)) for x in train_data]
    W_diag = np.diag(W)
    coef = np.ones((phi_x.shape[0],))
    for i in range(20):  # train phase: Newton-Raphson method for finding coef
        model = np.matmul(np.transpose(coef), phi_x)
        logistic_model = np.array([sigmoid(m) for m in model], dtype='float64')
        error = label - logistic_model
        gradient = coef * (-1 * landa) + np.matmul(phi_x, np.matmul(W_diag, error))
        # s(i,i)= w(i) * sigmoid(i)*(1-sigmoid(i))
        s_diag = np.diag([item[0] * item[1] * (1 - item[1]) for item in zip(W, logistic_model)])
        hessian = np.identity(len(coef)) * (-1 * landa) - np.matmul(np.matmul(phi_x, s_diag), np.transpose(phi_x))
        coef = coef - np.matmul(np.linalg.inv(hessian), gradient)
    return coef


def plot(feature1, feature2, x_predicted, y_predicted, predicted_model, actual_label, tau):
    trace1 = go.Scatter(
        name='real',
        x=feature1,
        y=feature2,
        mode='markers',
        marker=dict(color=actual_label,
                    colorscale=[[0, 'rgb(39,0,145)'], [1, 'rgb(225,0,30)']],
                    )
    )

    trace2 = go.Contour(
        x=x_predicted,
        y=y_predicted,
        z=predicted_model,
        showlegend=True,
        colorscale=[[0, 'rgb(194,226,253)'], [1, 'rgb(255,215,7)']],
        # ncontours=2,
    )

    fig = go.Figure(data=[trace2, trace1], layout_title_text="tau=" + str(tau))
    fig.show()


def create_model(coef, feature1, feature2):
    predicted_model = np.zeros((len(feature1), len(feature2)))
    for i in range(len(feature1)):
        for j in range(len(feature2)):
            predicted_label = np.matmul(np.transpose(coef), phi([[feature1[i], feature2[j]]], option=2))
            predicted_model[i][j] = sigmoid(predicted_label)
    return predicted_model


tau = [0.01, 0.05, 0.8, 0.1, 0.5, 1, 5]
x_train = read_txt_as_float("features.txt", ' ')
label_train = read_txt_as_float("labels.txt", ' ').T[0]

phi_train = phi(x_train, option=2)

test_feature1 = np.linspace(-1, 1, 200)
test_feature2 = np.linspace(-1, 1, 200)

predicted_model_matrix = np.zeros((200, 200))

for t in tau:
    for i in range(0, 200, 10):
        for j in range(0, 200, 10):
            # make grading in test data
            x1 = test_feature1[i:i + 10]
            x2 = test_feature2[j:j + 10]
            data_point = np.array([x1[int(len(x1) / 2)], x2[int(len(x2) / 2)]])
            coef = train(x_train, data_point, label_train,
                         phi_train, landa=0.0001, tau=t)
            # apply the model to the test data
            predicted_m = create_model(coef, x1, x2)
            predicted_model_matrix[i:i + 10, j:j + 10] = predicted_m
    # visualising our model
    plot(feature1=x_train.T[0], feature2=x_train.T[1], x_predicted=test_feature1,
         y_predicted=test_feature2,
         predicted_model=np.transpose(predicted_model_matrix), actual_label=label_train,
         tau=t)
