import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from PrepareData import PrepareData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def train(data, target_col):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_alpha = 0.01
    model = Lasso(alpha=best_alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    return X_train, X_test, y_train, y_test


def lasso_analyze(X_train, y_train):
    alphas = np.logspace(-4, 2, 100)
    coefs = []
    errors = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)
        errors.append(np.sqrt(mean_squared_error(y_train, lasso.predict(X_train))))

    plt.semilogx(alphas, errors)
    plt.show()

    best_alpha = alphas[np.argmin(errors)]
    best_lasso = Lasso(alpha=best_alpha)
    best_lasso.fit(X_train, y_train)

    most_important_idx = np.argmax(np.abs(best_lasso.coef_))
    most_important_feature = X_train.columns[most_important_idx]
    print(f"Most important feature: {most_important_feature}")

    return best_lasso


def main():
    file_path = r"C:\Dropbox\main1\Java\oris\02_homework\src\main\java\ru\itis\AmesHousing.csv"

    data = pd.read_csv(file_path)

    PD = PrepareData()

    prepared_data = PD.preprocess(data)
    print(prepared_data.columns)
    PD.plot_3d(prepared_data, 'SalePrice')

    X_train, X_test, y_train, y_test = train(prepared_data, 'SalePrice')

    lasso = lasso_analyze(X_train, y_train)
    print(lasso)

if __name__ == "__main__":
    main()