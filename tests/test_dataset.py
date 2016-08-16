from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston

from heamy.dataset import Dataset
import numpy as np
import pytest
import pandas as pd
import os
import shutil

from heamy.utils import flush_cache

np.random.seed(1000)

# X_train = np.random.rand(100, 100)
# X_test = np.random.rand(100, 100)
# y_train = np.random.rand(100, 1)
# y_test = np.random.rand(100, 1)
flush_cache()

data = load_boston()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)


class TestDataset(Dataset):
    @staticmethod
    def preprocess():
        data = load_boston()
        X, y = data['data'], data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
        return X_train, y_train, X_test, y_test


class TestDataset2(Dataset):
    @staticmethod
    def preprocess():
        return X_train, y_train


class TestDataset3(Dataset):
    @staticmethod
    def preprocess():
        return pd.DataFrame(X_train), pd.DataFrame(y_train)


class TestDataset4(Dataset):
    @staticmethod
    def preprocess():
        return {'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                }


def preprocess():
    return X_train, y_train, X_test, y_test


def test_initialization():
    Dataset(X_train, y_train, X_test, y_test)
    Dataset(X_train, y_train, X_test)
    Dataset(X_train, y_train)

    Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test))

    # with pytest.raises(ValueError):
    #     Dataset()

    Dataset(preprocessor=preprocess)

    TestDataset()


def test_properties():
    d = Dataset(X_train, y_train, X_test, y_test)
    assert d.X_train is X_train
    assert d.y_train is y_train
    assert d.X_test is X_test
    assert d.y_test is y_test

    d = Dataset(X_train, y_train)

    assert d.X_test is None
    assert d.y_test is None


def test_hashing():
    assert Dataset(X_train, y_train, X_test, y_test).hash == '0cb7e710b7319bb71e7328e4b422b374'
    assert Dataset(X_train, y_train, X_test).hash == 'c9b316f827981b3d0b53f8ab139234ea'
    assert Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train),
                   pd.DataFrame(X_test)).hash == 'c9b316f827981b3d0b53f8ab139234ea'

    assert Dataset(np.asfortranarray(X_train), np.asfortranarray(y_train),
                   np.asfortranarray(X_test)).hash == '8087697aa8460a25314edc85cc915ec8'

    d_hash = TestDataset().hash
    assert d_hash == TestDataset().hash


def test_repr():
    assert str(Dataset(X_train, y_train, X_test, y_test)) == 'Dataset(0cb7e710b7319bb71e7328e4b422b374)'


def test_shapes():
    x_t = np.random.rand(100, 5)
    y_t = np.random.rand(5, 1)

    with pytest.raises(ValueError):
        assert Dataset(X_train, y_train, x_t, y_test)

    with pytest.raises(ValueError):
        assert Dataset(X_train, y_t, x_t, y_test)

    with pytest.raises(ValueError):
        assert Dataset(X_train, y_train, X_test, y_t)

    with pytest.raises(ValueError):
        assert Dataset(X_train, y_train, x_t)


def test_preprocessing():
    TestDataset()
    TestDataset2()

    TestDataset4()


def test_split():
    d = Dataset(X_train, y_train)
    d.split(inplace=True)
    assert d.X_test is not None
    assert d.y_test is not None

    d = Dataset(X_train, y_train)
    data = d.split(inplace=False)
    assert all(x is not None for x in data)

    d = Dataset(X_train, y_train)
    data = d.split(inplace=False)
    assert all([x is not None for x in data])

    d = Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train))
    train_ind = np.array(range(0, 70))
    test_ind = np.array(range(30, 100))
    data = d.split(indices=[train_ind, test_ind])
    assert isinstance(data[0], (pd.DataFrame, pd.Series))
    assert isinstance(data[2], (pd.DataFrame, pd.Series))


def test_cache():
    d = Dataset(X_train, y_train, X_test, y_test)
    assert not d._cache()

    cache_dir = '.cache/heamy/'

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    d = TestDataset()
    d._cache()

    d = TestDataset(use_cache=True)
    d.load()
    assert d.loaded

    d = TestDataset(use_cache=False)
    d.load()
    assert d.loaded

    d = TestDataset3(use_cache=True)
    d.load()
    d._cache()

    d = TestDataset3(use_cache=True)
    d.load()
    assert isinstance(d.X_train, (pd.DataFrame, pd.Series))


def test_slicing():
    train_index = np.array(range(100))
    test_index = np.array(range(100, 250))

    d = Dataset(X_train, y_train, X_test, y_test)

    Xtrain, ytrain, Xtest, ytest = d.split(indices=(train_index, test_index))
    assert Xtrain.shape[0] == 100
    assert ytrain.shape[0] == Xtrain.shape[0]

    assert Xtest.shape[0] == 150
    assert Xtest.shape[0] == ytest.shape[0]

    d = Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test))
    Xtrain, ytrain, Xtest, ytest = d.split(indices=(train_index, test_index))
    assert Xtrain.shape[0] == 100
    assert ytrain.shape[0] == Xtrain.shape[0]

    assert Xtest.shape[0] == 150
    assert Xtest.shape[0] == ytest.shape[0]


def test_merge():
    d1 = Dataset(X_train, y_train, X_test, y_test)
    d2 = Dataset(X_train, y_train, X_test, y_test)
    ds = d1.merge(d2)
    assert ds.X_train.shape[1] == 26
    assert ds.X_test.shape[1] == 26

    d1 = Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test))
    d2 = Dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test))
    d1.merge(d2, inplace=True)
    assert d1.X_train.shape[1] == 26
    assert d1.X_test.shape[1] == 26


def test_convertion():
    d1 = Dataset(X_train, y_train, X_test, y_test)
    d1.to_csc()
    d1.to_dense()

    d1 = Dataset(X_train, y_train, X_test, y_test)
    d1.to_csr()
    d1.to_dense()

    d1 = Dataset(X_train, y_train, X_test, y_test)
    d1.to_dense()

