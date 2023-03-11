import numpy as np
import plotly.graph_objects as go


def read_txt_as_float(path, separator=" "):
    ls = []
    with open(path, 'r') as file:
        for line in file:
            data = list(filter(lambda x: x != '', line.split(separator)))
            ls.append(list(map(lambda x: float(x), data)))
    return np.array(ls)


def train(t_data, t_label, w=5):
    phi = [[x[0] ** i for i in range(0, w + 1)] for x in t_data]
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.transpose(phi)), t_label)


def predict(t_data, coefficient):
    coef = np.array(coefficient)
    phi = np.array([[x[0] ** i for i in range(0, coef.shape[0])] for x in t_data]).T
    return np.matmul(np.transpose(coef), phi).T


def plot(data, label, predicted):
    polynom = np.poly1d(np.polyfit(data, predicted, 5))
    y_p = polynom(np.linspace(-5, 13))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data, y=label,
                             mode='markers',
                             name='Data'))
    fig.add_trace(go.Scatter(x=np.linspace(-5, 13), y=y_p,
                             mode='lines',
                             name='Predicted'))
    fig.show()


data = read_txt_as_float("features.txt")
label = read_txt_as_float("labels.txt")

train_data = data[30:]
test_data = data[:30]

train_label = label[30:]
test_label = label[:30]

coef = train(train_data, train_label)
predicted = predict(test_data, coef)
predicted_t = predict(train_data, coef)

plot([x[0] for x in test_data.tolist()],
     [x[0] for x in test_label.tolist()],
     [x[0] for x in predicted.tolist()])

plot([x[0] for x in train_data.tolist()],
     [x[0] for x in train_label.tolist()],
     [x[0] for x in predicted_t.tolist()])

print("Error:",sum([(test_label[i]-predicted[i])**2 for i in range(0,len(predicted))]))
