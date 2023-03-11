"""
Logistic Regression
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


def train(label, landa, phi):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    coef = np.ones((phi.shape[0],))
    for i in range(10):  # train phase: Newton-Raphson method for finding coef
        model = np.matmul(np.transpose(coef), phi)
        logistic_model = np.array([sigmoid(m) for m in model], dtype='float64')
        error =label - logistic_model
        gradient = coef * (-1 * landa) + np.matmul(phi, error)
        s_diag = np.diag([s * (1 - s) for s in logistic_model])
        hessian = np.identity(len(coef)) * (-1 * landa) - np.matmul(np.matmul(phi, s_diag), np.transpose(phi))
        coef = coef - np.matmul(np.linalg.inv(hessian), gradient)

    return coef


def predict(coef, phi):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    model = np.matmul(np.transpose(coef), phi)
    logistic_model = np.array([sigmoid(m) for m in model], dtype='float64')
    predicted = [round(x) if abs(x - 0.5) > 0.01 else 0.5 for x in logistic_model]
    return predicted


def plot(x, y, predicted_label, actual_label):
    trace1 = go.Scatter(
        name='real',
        x=x,
        y=y,
        mode='markers',
        marker=dict(color=actual_label,
                    colorscale=[[0, 'rgb(39,0,145)'], [1, 'rgb(225,0,30)']],
                    )
    )

    trace2 = go.Contour(
        x=x,
        y=y,
        z=predicted_label,
        showlegend=True,
        colorscale=[[0, 'rgb(194,226,253)'], [1, 'rgb(255,215,7)']],
        ncontours=2,
    )

    fig = go.Figure(data=[trace2, trace1])
    fig.show()


x_train = read_txt_as_float("features.txt", ' ')
label_train = read_txt_as_float("labels.txt", ' ').T[0]
# phi = np.transpose([[1, data[0], data[1] ** 2] for data in x_train])
# phi = np.transpose([[1, data[0], data[1], data[0] * data[1], data[0] ** 2 * data[1]]
#                   for data in x_train])
phi = np.transpose([[1, data[0], data[1], data[0] * data[1], data[0] ** 2 * data[1], data[0] * data[1] ** 2]
                    for data in x_train])

coefficient = train(label_train, landa=0.0001, phi=phi)
predicted = predict(coefficient, phi)  # apply the model to the train data
plot(x=x_train.T[0], y=x_train.T[1], predicted_label=np.transpose(predicted), actual_label=label_train)  # visualising our model
