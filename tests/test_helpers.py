# coding:utf-8
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from heamy.cache import np_hash
from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.utils.main import generate_columns, group_models, report_score
from heamy.utils.optimizer import Optimizer


def boston_dataset():
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
    return X_train, y_train, X_test, y_test


dataset = Dataset(preprocessor=boston_dataset, use_cache=True)
model = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True})
model_2 = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50})


def test_generate_columns():
    x = np.random.rand(100, 10)
    output = generate_columns(x, 'test')
    assert len(output) == 10
    assert all([True if col.find('test') == 0 else False for col in output])

    output = generate_columns(np.random.rand(100), 'test')
    assert output[0] == 'test'


def test_optimizer():
    opt = Optimizer([model, model_2], scorer=mean_absolute_error)
    output = opt.minimize('SLSQP')
    assert output.shape[0] == 2
    assert_almost_equal(output.sum(), 1.0, decimal=5)


def test_group_models():
    y_pred, y_true = group_models([model, model_2], {'k': 5})
    assert set(y_pred.keys()) == set(y_true.keys())
    assert len(y_pred[0]) == 2


def test_report_score():
    report_score(np.array([1, 2, 3]), mean_absolute_error)
    report_score(np.array([1, 2, 3]), None)


def test_np_hash():
    np.random.seed(9999)
    x = np.random.rand(100, 100)
    assert np_hash(x) == '44ea3be8b8ddc5af8e3c2a01ba5c6be6'
