# coding:utf-8
import random

import numpy as np
import pytest
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression

from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier

np.random.seed(1000)
random.seed(1111)

X_train = np.random.rand(10, 100)
X_test = np.random.rand(10, 100)
y_train = np.random.rand(10, 1)
y_test = np.random.rand(10, 1)


class RealDataset(Dataset):
    @staticmethod
    def preprocess():
        data = load_boston()
        X, y = data['data'], data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}


class TestDataset(Dataset):
    @staticmethod
    def preprocess():
        return X_train, y_train, X_test, y_test


class TestEstimator(Classifier):
    @staticmethod
    def estimator(X_train, y_train, X_test, y_test=None):
        return np.zeros((2, X_test.shape[0]))
        # return np.zeros(X_test.shape[0])


def func_estimator(X_train, y_train, X_test, y_test):
    return np.zeros(X_test.shape[0])


def random_param():
    return random.randint(1, 100)


model_func = Regressor(estimator=func_estimator, dataset=TestDataset)
model_cls = TestEstimator(dataset=TestDataset())
model_param = Regressor(estimator=LinearRegression, parameters={'random_param': random_param}, dataset=TestDataset)
model_param2 = Classifier(estimator=LogisticRegression, parameters={'colsample_bylevel': 0.9}, dataset=TestDataset)


def test_hashing():
    assert str(model_func) == 'func_estimator(54743c7a5484d1bf2a64ac1d7b68f8cc)'
    assert str(model_cls) == 'TestEstimator(da29cb8766f96e6561a51e8e3c13f661)'
    assert str(model_param) == 'LinearRegression(2e789a766f6dc2457fb6a63452ad2859)'
    assert str(model_param2) == 'LogisticRegression(74efb248db47d168aed2fc37c0016e6f)'

    assert model_param2.hash == '74efb248db47d168aed2fc37c0016e6f'

    e_hash = TestEstimator(dataset=TestDataset()).hash
    assert e_hash == TestEstimator(dataset=TestDataset()).hash


def test_custom_estimators():
    def test_estimator():
        return

    class TestEstimator2(Classifier):
        def estimator(self):
            return

    with pytest.raises(ValueError):
        TestEstimator2(dataset=TestDataset)

    with pytest.raises(ValueError):
        Regressor(estimator=test_estimator, dataset=TestDataset)


def test_validation():
    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    model.validate(k=10)
    # Retrieve cached object
    y_true, y_pred = model.validate(k=10)
    assert len(y_true) == len(y_pred)

    model.validate(k=1)
    # Retrieve cached object
    y_true, y_pred = model.validate(k=1)
    assert len(y_true) == len(y_pred)
    assert len(y_true) == 1


def test_prediction():
    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    output = model.predict()
    assert len(output.shape) == 1
    assert model.dataset.X_test.shape[0] == output.shape[0]

    # Retrieve cached object
    output = model.predict()
    assert len(output.shape) == 1
    assert model.dataset.X_test.shape[0] == output.shape[0]


def test_stacking():
    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    ds = model.stack(10)

    assert ds.X_train.shape[0] == model.dataset.X_train.shape[0]
    assert ds.X_test.shape[0] == model.dataset.X_test.shape[0]
    assert ds.y_train.shape[0] == model.dataset.y_train.shape[0]

    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    ds = model.stack(10, full_test=False)
    assert np.isnan(ds.X_train).sum() == 0
    assert ds.X_train.shape[0] == model.dataset.X_train.shape[0]
    assert ds.X_test.shape[0] == model.dataset.X_test.shape[0]
    assert ds.y_train.shape[0] == model.dataset.y_train.shape[0]

    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    model.dataset.load()
    ds = model.stack(10, full_test=False)
    # Check cache
    assert np.isnan(ds.X_train).sum() == 0
    assert ds.X_train.shape[0] == model.dataset.X_train.shape[0]
    assert ds.X_test.shape[0] == model.dataset.X_test.shape[0]
    assert ds.y_train.shape[0] == model.dataset.y_train.shape[0]


def test_blending():
    model = Regressor(estimator=LinearRegression, parameters={}, dataset=RealDataset)
    _, _, X_t, y_t = model.dataset.split(test_size=0.2)
    ds = model.blend(proportion=0.2)
    assert ds.X_test.shape[0] == model.dataset.X_test.shape[0]
    assert ds.X_train.shape[0] == X_t.shape[0]

    # Check cache
    ds = model.blend(proportion=0.2)
    assert ds.X_test.shape[0] == model.dataset.X_test.shape[0]
    assert ds.X_train.shape[0] == X_t.shape[0]
