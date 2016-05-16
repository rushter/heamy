# coding:utf-8
import pandas as pd


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

        train[column] = train[column].astype('category', categories=categories)
        test[column] = test[column].astype('category', categories=categories)

    train = pd.get_dummies(train, columns=features, dummy_na=dummy_na, sparse=sparse)
    test = pd.get_dummies(test, columns=features, dummy_na=dummy_na, sparse=sparse)

    # d_cols = train.columns[(train == 0).all()]
    # train.drop(d_cols, 1, inplace=True)
    # test.drop(d_cols, 1, inplace=True)

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
