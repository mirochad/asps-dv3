"""Shared fixed data splits, representation and coalition utility."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticGame:
    def __init__(self, seed=0):
        self.seed = seed
        self.data = load_breast_cancer()
        ids = np.arange(len(self.data.target))
        pool, self.test = train_test_split(ids, test_size=0.2, stratify=self.data.target, random_state=seed)
        pool, self.validation = train_test_split(pool, test_size=0.25, stratify=self.data.target[pool], random_state=seed)
        self.train, _ = train_test_split(pool, train_size=12, stratify=self.data.target[pool], random_state=seed)
        self.scaler = StandardScaler().fit(self.data.data[self.train])
        self.x = self.scaler.transform(self.data.data[self.train])
        self.x_val = self.scaler.transform(self.data.data[self.validation])
        self.y = self.data.target[self.train]
        self.y_val = self.data.target[self.validation]
        self.n = len(self.train)
        self.calls = 0
        self.fits = 0

    def model(self):
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=self.seed)

    def __call__(self, mask):
        self.calls += 1
        if mask == 0:
            return 0.0
        subset = [i for i in range(self.n) if mask & (1 << i)]
        labels = self.y[subset]
        if len(np.unique(labels)) == 1:
            return float(np.mean(self.y_val == labels[0]))
        model = self.model().fit(self.x[subset], labels)
        self.fits += 1
        return float(np.mean(model.predict(self.x_val) == self.y_val))

    def splits(self):
        return dict(train=self.train.tolist(), validation=self.validation.tolist(), test=self.test.tolist())
