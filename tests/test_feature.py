# coding:utf-8
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris

from heamy.feature import onehot_features, factorize, woe, mean_target


def test_onehot():
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=333)
    train = pd.DataFrame(X_train)
    test = pd.DataFrame(X_test)

    t_train, t_test = onehot_features(train.copy(deep=True), test.copy(deep=True), [8, 1, 12], full=False,
                                      dummy_na=True)
    assert t_train.shape[1] == t_test.shape[1]
    assert t_train.shape[1] == 441

    t_train, t_test = onehot_features(train.copy(deep=True), test.copy(deep=True), [8, 1, 12], full=True,
                                      dummy_na=False)
    assert t_train.shape[1] == t_test.shape[1]
    assert t_train.shape[1] == 500


def test_factorize():
    train = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': ['a', 'b', 'c']})
    test = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': ['z', 'b', 'c']})

    t_train, t_test = factorize(train.copy(deep=True), test.copy(deep=True), ['a', 'b'], full=True, )
    assert len(t_train.select_dtypes(include=[np.int64, np.int32]).columns) == 2
    assert len(t_test.select_dtypes(include=[np.int64, np.int32]).columns) == 2
    assert t_test['b'].nunique() == 3

    t_train, t_test = factorize(train.copy(deep=True), test.copy(deep=True), ['a', 'b'], full=False, na_value=np.nan)
    assert t_test['b'].dropna().nunique() == 2


def test_target_transformations():
    X = pd.DataFrame(np.random.randint(2, size=(100, 2)), columns=('x', 'target'))
    output = woe(X, 'x', 'target')
    assert output.shape[0] == 100

    output = mean_target(X, 'x', 'target', C=10)
    assert output.shape[0] == 100

    output = mean_target(X, 'x', 'target', C=None)
    assert output.shape[0] == 100
