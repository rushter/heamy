# coding:utf-8
import os
import shutil
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from six.moves import range

try:
    import tqdm as tqdm
except:
    pass


def report_score(scores, metric=None):
    if metric is not None:
        print('Metric: %s' % metric.__name__)
    if len(scores) == 1:
        print('Accuracy: %s' % scores[0])
    else:
        print('Folds accuracy: %s' % scores)
        print('Mean accuracy: %s' % np.mean(scores))
        print('Standard Deviation: %s' % np.std(scores))
        print('Variance: %s' % np.var(scores))


def tsplit(df, shape):
    """Split array into two parts."""
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.iloc[0:shape], df.iloc[shape:]
    else:
        return df[0:shape], df[shape:]


def concat(x, y, axis=0):
    """Concatenate a sequence of pandas or numpy objects into one entity."""
    if all([isinstance(df, (pd.DataFrame, pd.Series)) for df in [x, y]]):
        return pd.concat([x, y], axis=axis)
    else:
        if axis == 0:
            return np.concatenate([x, y])
        else:
            return np.column_stack([x, y])


def reshape_1d(df):
    """If parameter is 1D row vector then convert it into 2D matrix."""
    shape = df.shape
    if len(shape) == 1:
        return df.reshape(shape[0], 1)
    else:
        return df


def idx(df, index):
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.iloc[index]
    else:
        return df[index, :]


def generate_columns(df, name):
    if len(df.shape) == 1:
        col_count = 1
    else:
        col_count = df.shape[1]
    if col_count == 1:
        return [name]
    else:
        return ['%s_%s' % (name, i) for i in range(col_count)]


def group_models(models, params):
    y_preds_grouped = defaultdict(list)
    y_true_grouped = {}
    for model in models:
        y_true_list, y_pred_list = model.validate(**params)
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            if i not in y_true_grouped:
                y_true_grouped[i] = y_true
            y_preds_grouped[i].append(y_pred)
    return y_preds_grouped, y_true_grouped


def feature_combiner(df):
    combs = list(combinations(df.columns, 2))
    for i, j in combs:
        column_name = '%s-%s' % (i, j)
        df[column_name] = df[i] - df[j]
    return df


def flush_cache():
    cache_dir = '.cache/heamy/'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def xgb_progressbar(rounds=1000):
    """Progressbar for xgboost using tqdm library.

    Examples
    --------

    >>> model = xgb.train(params, X_train, 1000, callbacks=[xgb_progress(1000), ])
    """
    pbar = tqdm(total=rounds)

    def callback(_, ):
        pbar.update(1)

    return callback
