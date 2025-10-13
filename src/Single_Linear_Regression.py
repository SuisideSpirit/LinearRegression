import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def loss_function(slope , inter , df):
    n = len(df)
    loss = 0 
    for i in range(n):
        x = df.iloc[i, 0]
        y = df.iloc[i, 1]
        loss += (y - (slope*x - inter))**2
    return loss/n

def gradient_descent(df, learning_rate , slope , inter):
    n = len(df)
    slope_deri = 0 
    inter_deri = 0
    for i in range(n):
        x = df.iloc[i,0]
        y = df.iloc[i,1]
        slope_deri += (-2*x)*(y - (slope*x + inter))
        inter_deri += -2*(y - (slope*x + inter))

    slope = slope - (learning_rate * (slope_deri / n))
    inter = inter - (learning_rate * (inter_deri / n))
    return slope , inter

def train_linear_regression(df, learning_rate=0.001, epochs=300, verbose=True):
    slope, inter = 0, 0

    for i in range(epochs + 1):
        slope, inter = gradient_descent(df, learning_rate, slope, inter)

    return slope, inter


def plot_regression(df, slope, inter):
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color='blue', label='Data')
    plt.plot(df.iloc[:, 0], slope * df.iloc[:, 0] + inter, color='red', label='Regression Line')
    plt.legend()
    plt.show()

def predict(slope, inter, x):
    n = len(x)
    y = []
    for i in range(n):
        y.append(slope * x[i] + inter)
    return y

if __name__ == "__main__":
    df = pd.read_csv('Linear_Regression\data\score.csv')
    slope, inter = train_linear_regression(df, learning_rate=0.001, epochs=300)
    print(f"Slope: {slope}, Intercept: {inter}")
    plot_regression(df, slope, inter)
    x_test = [5, 10, 15]
    predictions = predict(slope, inter, x_test)
    print(f"Predictions for {x_test}: {predictions}")