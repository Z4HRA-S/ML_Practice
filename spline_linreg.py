import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def read_txt_as_float(path, separator=" "):
    ls = []
    with open(path, 'r') as file:
        for line in file:
            data = list(filter(lambda x: x != '', line.split(separator)))
            ls.append(list(map(lambda x: float(x), data)))
    return np.array(ls)


def phi(data, knot, p):
    phi_matrix = [[x[0] ** i for i in range(0, p + 1)] for x in data]
    positive = lambda x: (abs(x) + x) / 2
    phi_matrix2 = [[positive(x[0] - knot[k]) ** p for k in range(0, len(knot) - 1)] for x in data]
    return np.hstack((phi_matrix, phi_matrix2))


def penalized_spline_train(phi_matrix, label, landa):
    coef = np.matmul(
        np.linalg.inv(np.matmul(np.transpose(phi_matrix), phi_matrix) + (landa * np.identity(phi_matrix.shape[1]))),
        np.matmul(np.transpose(phi_matrix), label))
    return coef


def penalized_spline_test(phi_matrix, coef):
    return np.transpose(np.matmul(np.transpose(coef), np.transpose(phi_matrix)))


def knot(data, q):
    min = np.min(data)
    max = np.max(data)
    period = (max - min) / q
    return [min + (i * period) for i in range(0, q)]


def plot_test(data, label, predicted, p, q):
    fig = go.Figure(layout_title_text="p=" + str(p) + ", q=" + str(q))
    fig.add_trace(go.Scatter(x=data, y=label,
                             mode='markers',
                             name='Data'))
    fig.add_trace(go.Scatter(x=data, y=predicted,
                             mode='markers',
                             name='Predicted'))
    fig.show()


'''def plot_fit_linear(coef, knot, data, label):
    # sort data and corresponding labels and predicted
    data, label = np.transpose(data), np.transpose(label)
    idx = np.argsort(data)
    data = data[idx]
    label = label[idx]
    # sorted!
    period = np.floor((np.max(data) - np.min(data)) / len(knot))
    knot.insert(0, 0)  # begining knot of index 1
    line_slope = coef[1]
    line_intercept = coef[0]
    for i in range(1, len(knot)):
        line_slope = line_slope + coef[i + 1]
        line_intercept = line_intercept - coef[i + 1] * knot[i]
        line_fit = np.poly1d([line_slope, line_intercept])
        domain = np.linspace(data[i], data[i] + period)
        plt.plot(domain, line_fit(domain), c="red")
        plt.scatter(data[i * period:(i + 1) * period], label[i * period:(i + 1) * period], c="green")
        plt.show()
'''


def plot_fit_line(data, label, predicted, knot):
    # sort data and corresponding labels and predicted
    data, label, predicted = np.transpose(data), np.transpose(label), np.transpose(predicted)
    idx = np.argsort(data)
    data = data[idx]
    predicted = predicted[idx]
    label = label[idx]
    # sorted!
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data, y=label,
                             mode='markers',
                             name='Data'))
    fig.add_trace(go.Scatter(x=data, y=predicted,
                             mode='lines',
                             name='Predicted',
                             ))
    fig.show()


q = 6
p = 1
landa = 0.5
data = read_txt_as_float("features.txt")
label = read_txt_as_float("labels.txt")

train_data = data[30:]
test_data = data[:30]
train_label = label[30:]
test_label = label[:30]
phi_matrix_train = phi(train_data, knot(train_data, q), p)
coef = penalized_spline_train(phi_matrix_train, train_label, landa)

phi_matrix_test = phi(test_data, knot(train_data, q), p)
predicted = penalized_spline_test(phi_matrix_test, coef)

plot_test([x[0] for x in test_data],
          [x[0] for x in test_label],
          [x[0] for x in predicted], p, q)

phi_matrix_train = phi(train_data, knot(train_data, q), p)
predicted_train = penalized_spline_test(phi_matrix_train, coef)

plot_fit_line([x[0] for x in train_data],
              [x[0] for x in train_label],
              [x[0] for x in predicted_train],knot(train_data,q))

print("Error:", sum([(test_label[i] - predicted[i]) ** 2 for i in range(0, len(predicted))]))
