import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PrepareData:

    def preprocess(self, data):
        numeric_cols = data.select_dtypes(include=['number']).columns
        data = data[numeric_cols]

        corr = data.corr().abs()

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.81)]
        data = data.drop(to_drop, axis=1)

        data = data.dropna()

        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        print(data)
        return data

    def plot_3d(self, data, target_col):
        features = data.drop(target_col, axis=1)
        target = data[target_col]

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(reduced[:, 0], reduced[:, 1], target, c=target)
        ax.set_zlabel(target_col)
        plt.show()