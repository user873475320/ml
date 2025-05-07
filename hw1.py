import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Optional

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Нахождение оптимального количества кластеров по силуэту
# -----------------------------
def find_optimal_clusters(dataset: np.ndarray, max_clusters: int = 10) -> int:
    optimal_k = 2
    best_metric = -1
    metric_values = []

    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42)
        cluster_labels = model.fit_predict(dataset)
        score = silhouette_score(dataset, cluster_labels)
        metric_values.append(score)

        if score > best_metric:
            best_metric = score
            optimal_k = k

    return optimal_k

# -----------------------------
# 2. Класс CustomKMeans — собственная реализация алгоритма
# -----------------------------
class CustomKMeans:
    def __init__(self, data: Optional[np.ndarray] = None, clusters: int = 2):
        self.data = np.asarray(data) if data is not None else None
        self.k = clusters
        self.centers = []  # координаты центроидов
        self.cluster_assignments = []  # к какому кластеру принадлежит точка

    # Евклидово расстояние между двумя точками
    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    # Инициализация центров методом k-means++
    def _initialize(self) -> List[np.ndarray]:
        init_centers = []
        first = random.randint(0, len(self.data) - 1)
        init_centers.append(self.data[first])

        for _ in range(1, self.k):
            dist_list = []
            for p in self.data:
                min_dist = min(self._euclidean(p, c) for c in init_centers)
                dist_list.append(min_dist)

            total = sum(dist_list)
            probs = [d / total for d in dist_list]
            selected = np.random.choice(len(self.data), p=probs)
            init_centers.append(self.data[selected])

        return init_centers

    # Назначение точки ближайшему центру
    def _assign(self) -> List[int]:
        labels = []
        for d in self.data:
            dists = [self._euclidean(d, c) for c in self.centers]
            labels.append(np.argmin(dists))
        return labels

    # Пересчёт центров кластеров
    def _recompute(self) -> bool:
        updated = []
        for i in range(self.k):
            points = self.data[np.array(self.cluster_assignments) == i]
            if points.shape[0] == 0:
                updated.append(self.centers[i])  # если пусто, не менять центр
            else:
                updated.append(np.mean(points, axis=0))

        if np.allclose(self.centers, updated):
            return False  # центры почти не изменились, можно остановиться

        self.centers = updated
        return True

    # Обучение модели
    def train(self, max_iters: int = 100):
        self.centers = self._initialize()

        for _ in range(max_iters):
            self.cluster_assignments = self._assign()
            self._plot_step()
            if not self._recompute():
                break

    # Предсказание кластера для новых точек
    def predict(self, points: List[List[float]]) -> List[int]:
        points = np.array(points)
        preds = []
        for p in points:
            distances = [self._euclidean(p, c) for c in self.centers]
            preds.append(np.argmin(distances))
        return preds

    # Отрисовка одного шага (в 2D проекции)
    def _plot_step(self, dim1: int = 0, dim2: int = 1):
        plt.scatter(self.data[:, dim1], self.data[:, dim2], c=self.cluster_assignments, cmap='plasma')
        centers_np = np.array(self.centers)
        plt.scatter(centers_np[:, dim1], centers_np[:, dim2], s=180, c='black', marker='x')
        plt.pause(0.1)
        plt.clf()

    # Финальная визуализация всех 2D-проекций (попарно)
    def plot_all_pairs(self):
        dims = self.data.shape[1]
        for i in range(dims):
            for j in range(dims):
                if i == j:
                    continue
                plt.subplot(dims, dims, i * dims + j + 1)
                plt.scatter(self.data[:, j], self.data[:, i], c=self.cluster_assignments, cmap='viridis', alpha=0.75)
                c_np = np.array(self.centers)
                plt.scatter(c_np[:, j], c_np[:, i], c='red', marker='x', s=100)
        plt.tight_layout()
        plt.show()

# -----------------------------
# 3. Основной запуск
# -----------------------------
def run():
    dataset = load_iris().data
    best_k = find_optimal_clusters(dataset)  # Находим оптимальное k

    model = CustomKMeans(data=dataset, clusters=best_k)
    model.train()  # Обучение с визуализацией шагов
    model.plot_all_pairs()  # Финальная визуализация

if __name__ == "__main__":
    run()
