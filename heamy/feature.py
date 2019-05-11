# coding:utf-8
import logging

import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder


def onehot_features(train, test, features, full=False, sparse=False, dummy_na=True):
    """Encode categorical features using a one-hot scheme.

    Parameters
    ----------
    train : pd.DataFrame
    test : pd.DataFrame
    features : list
        Column names in the DataFrame to be encoded.
    full : bool, default False
        Whether use all columns from train/test or only from train.
    sparse : bool, default False
        Whether the dummy columns should be sparse or not.
    dummy_na : bool, default True
        Add a column to indicate NaNs, if False NaNs are ignored.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """

    features = [f for f in features if f in train.columns]
    for column in features:
        if full:
            categories = pd.concat([train[column], test[column]]).dropna().unique()
        else:
            categories = train[column].dropna().unique()

        train[column] = train[column].astype("category", categories=categories)
        test[column] = test[column].astype("category", categories=categories)

    train = pd.get_dummies(train, columns=features, dummy_na=dummy_na, sparse=sparse)
    test = pd.get_dummies(test, columns=features, dummy_na=dummy_na, sparse=sparse)
    return train, test


def factorize(train, test, features, na_value=-9999, full=False, sort=True):
    """Factorize categorical features.

    Parameters
    ----------
    train : pd.DataFrame
    test : pd.DataFrame
    features : list
           Column names in the DataFrame to be encoded.
    na_value : int, default -9999
    full : bool, default False
        Whether use all columns from train/test or only from train.
    sort : bool, default True
        Sort by values.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """

    for column in features:
        if full:
            vs = pd.concat([train[column], test[column]])
            labels, indexer = pd.factorize(vs, sort=sort)
        else:
            labels, indexer = pd.factorize(train[column], sort=sort)

        train[column] = indexer.get_indexer(train[column])
        test[column] = indexer.get_indexer(test[column])

        if na_value != -1:
            train[column] = train[column].replace(-1, na_value)
            test[column] = test[column].replace(-1, na_value)

    return train, test


def woe(df, feature_name, target_name):
    """Calculate weight of evidence.

    Parameters
    ----------
    df: Dataframe
    feature_name: str
        Column name to encode.
    target_name: str
        Target column name.

    Returns
    -------
    Series

    """

    def group_woe(group):
        event = float(group.sum())
        non_event = group.shape[0] - event

        rel_event = event / event_total
        rel_non_event = non_event / non_event_total

        return np.log(rel_non_event / rel_event) * 100

    if df[target_name].nunique() > 2:
        raise ValueError("Target column should be binary (1/0).")

    event_total = float(df[df[target_name] == 1.0].shape[0])
    non_event_total = float(df.shape[0] - event_total)

    woe_vals = df.groupby(feature_name)[target_name].transform(group_woe)
    return woe_vals


def mean_target(df, feature_name, target_name, C=None):
    """Mean target.

    Original idea: Stanislav Semenov

    Parameters
    ----------
    C : float, default None
        Regularization coefficient. The higher, the more conservative result.
        The optimal value lies between 10 and 50 depending on the data.
    feature_name : str
    target_name : str
    df: DataFrame

    Returns
    -------
    Series
    """

    def group_mean(group):
        group_size = float(group.shape[0])
        if C is None:
            return (group.mean() * group_size + global_mean) / group_size
        else:
            return (group.mean() * group_size + global_mean * C) / (group_size + C)

    global_mean = df[target_name].mean()
    return df.groupby(feature_name)[target_name].transform(group_mean)


def xgb_to_features(model, X_train, X_test):
    """Converts xgboost model into categorical features.

    Reference:
    "Practical Lessons from Predicting Clicks on Ads at Facebook"
    https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
    """
    import xgboost as xgb

    f_train = model.predict(xgb.DMatrix(X_train), pred_leaf=True)
    f_test = model.predict(xgb.DMatrix(X_test), pred_leaf=True)
    enc = OneHotEncoder()
    enc.fit(f_train)
    return enc.transform(f_train), enc.transform(f_test)


class XGBParser(object):
    def __init__(self):
        self.groups = set()

    def load_dump(self, path):

        dump = open(path, "r").read()
        search = re.finditer(r"\[f([0-9]*?)<(.*?)\]", dump, re.MULTILINE)
        for group in search:
            group = group.groups()
            idx, val = int(group[0]), float(group[1])
            self.groups.add((idx, val))

        logging.debug("Found %s splits.", (len(self.groups)))

    def transform(self, X):
        output = []
        for i, group in enumerate(self.groups):
            idx, val = group
            cond = csr_matrix((X[:, idx] < val).reshape((-1, 1))).astype(np.int8)
            output.append(cond)
        output = hstack(output, dtype=np.int8)
        return output
