import numpy as np
import plotly.graph_objects as go


def read_txt_as_float(path, separator=" "):
    ls = []
    with open(path, 'r') as file:
        for line in file:
            data = list(filter(lambda x: x != '', line.split(separator)))
            ls.append(list(map(lambda x: float(x), data)))
    return np.array(ls)


def predict(data, label, data_point, tau):
    w = np.diagflat([np.exp(-1 * ((data_point - x[0]) ** 2) / (2 * tau ** 2)) for x in data])
    XtwX = np.matmul(np.matmul(np.transpose(data), w), data)
    XtwT = np.matmul(np.matmul(np.transpose(data), w), label)
    tetha = np.matmul(np.linalg.inv(XtwX), XtwT)
    return data_point * tetha[0]


def plot(data, label, predicted, tau):
    # sort data and corresponding labels and predicted
    data, label, predicted = np.transpose(data), np.transpose(label), np.transpose(predicted)
    idx = np.argsort(data)
    data = data[idx]
    predicted = predicted[idx]
    label = label[idx]
    # sorted!

    fig = go.Figure(layout_title_text="tau=" + str(tau))
    fig.add_trace(go.Scatter(x=data, y=label,
                             mode='markers',
                             name='Data'))
    fig.add_trace(go.Scatter(x=data, y=predicted,
                             mode='lines',
                             name='Predicted'))
    fig.show()


tau = [0.8, 0.1, 0.3, 2, 10]
data = read_txt_as_float("features.txt")
label = read_txt_as_float("labels.txt")

train_data = data[30:]
test_data = data[:30]
train_label = label[30:]
test_label = label[:30]

for t in tau:
    predicted = [predict(train_data, train_label, x[0], t)[0] for x in test_data]
    plot([x[0] for x in test_data],
         [x[0] for x in test_label],
         predicted, t)
    print("Error with tau=", t, "=", sum([(test_label[i] - predicted[i]) ** 2 for i in range(0, len(predicted))]))
