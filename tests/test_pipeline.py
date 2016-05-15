# coding:utf-8
import pytest
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline

import numpy as np


def boston_dataset():
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
    return X_train, y_train, X_test, y_test


dataset = Dataset(preprocessor=boston_dataset, use_cache=True)
model = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True}, name='lr')
model_2 = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50}, name='rf')

pipeline = ModelsPipeline(model, model_2)


def test_apply():
    output = pipeline.apply(lambda x: np.mean(x, axis=0)).execute()
    assert output.shape[0] == dataset.X_test.shape[0]

    output = pipeline.apply(lambda x: np.mean(x, axis=0)).validate(scorer=mean_absolute_error, k=10)
    assert len(output) == 10


def test_simple_functions():
    assert dataset.X_test.shape[0] == pipeline.max().execute().shape[0]
    assert dataset.X_test.shape[0] == pipeline.mean().execute().shape[0]
    assert dataset.X_test.shape[0] == pipeline.gmean().execute().shape[0]
    assert dataset.X_test.shape[0] == pipeline.weight([0.5, 0.5]).execute().shape[0]


def test_add():
    with pytest.raises(ValueError):
        pipeline.add(object)


def test_stack_and_blend():
    ds = pipeline.stack(k=10)
    assert dataset.X_test.shape[0] == ds.X_test.shape[0]
    assert dataset.X_train.shape[0] == ds.X_train.shape[0]
    assert dataset.y_train.shape[0] == ds.y_train.shape[0]

    assert ds.X_train.shape[1] == ds.X_test.shape[1]
    assert ds.X_test.shape[1] == 2
    pipeline.blend()

    ds = pipeline.stack(k=10, add_diff=True)
    assert ds.X_train.shape[1] == 3
    assert ds.X_test.shape[1] == 3

    assert 'lr-rf' in ds.X_train.columns
    assert 'lr-rf' in ds.X_test.columns
