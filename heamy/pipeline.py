# coding:utf-8

import numpy as np
import pandas as pd
from scipy.stats import gmean

from .dataset import Dataset
from .estimator import Regressor, Classifier
from .utils.main import report_score, generate_columns, group_models, feature_combiner
from .utils.optimizer import Optimizer


class ModelsPipeline(object):
    """Combines sequence of models."""

    def __init__(self, *args):
        self.models = []

        for model in args:
            if isinstance(model, (Regressor, Classifier)):
                self.add(model)
            else:
                raise ValueError('Unrecognized estimator.')

    def add(self, model):
        """Adds a single model.

        Parameters
        ----------
        model : `Estimator`
        """
        if isinstance(model, (Regressor, Classifier)):
            self.models.append(model)
        else:
            raise ValueError('Unrecognized estimator.')

    def apply(self, func):
        """Applies function along models output.

        Parameters
        ----------
        func : function
            Arbitrary function with one argument.

        Returns
        -------
        `PipeApply`

        Examples
        --------

        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> pipeline.apply(lambda x: np.max(x,axis=0)).execute()
        """
        return PipeApply(function=func, models=self.models)

    def mean(self):
        """Returns the mean of the models predictions.

        Returns
        -------
        `PipeApply`

        Examples
        --------
        >>> # Execute
        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> pipeline.mean().execute()

        >>> # Validate
        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> pipeline.mean().validate()

        """
        return self.apply(lambda x: np.mean(x, axis=0))

    def gmean(self):
        """Returns the gmean of the models predictions.

        Returns
        -------
        `PipeApply`
        """
        return self.apply(lambda x: gmean(x, axis=0))

    def max(self):
        """Returns the max of the models predictions.

        Returns
        -------
        `PipeApply`
        """
        return self.apply(lambda x: np.max(x, axis=0))

    def min(self):
        """Returns the min of the models predictions.

        Returns
        -------
        `PipeApply`
        """
        return self.apply(lambda x: np.min(x, axis=0))

    def stack(self, k=5, stratify=False, shuffle=True, seed=100, full_test=True, add_diff=False):
        """Stacks sequence of models.

        Parameters
        ----------

        k : int, default 5
            Number of folds.
        stratify : bool, default False
        shuffle : bool, default True
        seed : int, default 100
        full_test : bool, default True
            If True then evaluate test dataset on the full data otherwise take the mean of every fold.
        add_diff : bool, default False

        Returns
        -------
        `DataFrame`

        Examples
        --------
        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> stack_ds = pipeline.stack(k=10, seed=111)
        """
        result_train = []
        result_test = []
        y = None

        for model in self.models:
            result = model.stack(k=k, stratify=stratify, shuffle=shuffle, seed=seed, full_test=full_test)
            train_df = pd.DataFrame(result.X_train, columns=generate_columns(result.X_train, model.name))
            test_df = pd.DataFrame(result.X_test, columns=generate_columns(result.X_test, model.name))

            result_train.append(train_df)
            result_test.append(test_df)
            if y is None:
                y = result.y_train
        result_train = pd.concat(result_train, axis=1)
        result_test = pd.concat(result_test, axis=1)

        if add_diff:
            result_train = feature_combiner(result_train)
            result_test = feature_combiner(result_test)

        ds = Dataset(X_train=result_train, y_train=y, X_test=result_test)

        return ds

    def blend(self, proportion=0.2, stratify=False, seed=100, indices=None, add_diff=False):
        """Blends sequence of models.

        Parameters
        ----------
        proportion : float, default 0.2
        stratify : bool, default False
        seed : int, default False
        indices : list(np.ndarray,np.ndarray), default None
            Two numpy arrays that contain indices for train/test slicing.
        add_diff : bool, default False

        Returns
        -------
        `DataFrame`

        Examples
        --------
        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> pipeline.blend(seed=15)

        >>> # Custom indices
        >>> train_index = np.array(range(250))
        >>> test_index = np.array(range(250,333))
        >>> res = model_rf.blend(indicies=(train_index,test_index))
        """
        result_train = []
        result_test = []
        y = None

        for model in self.models:
            result = model.blend(proportion=proportion, stratify=stratify, seed=seed, indices=indices)
            train_df = pd.DataFrame(result.X_train, columns=generate_columns(result.X_train, model.name))
            test_df = pd.DataFrame(result.X_test, columns=generate_columns(result.X_test, model.name))

            result_train.append(train_df)
            result_test.append(test_df)
            if y is None:
                y = result.y_train
        result_train = pd.concat(result_train, axis=1, ignore_index=True)
        result_test = pd.concat(result_test, axis=1, ignore_index=True)

        if add_diff:
            result_train = feature_combiner(result_train)
            result_test = feature_combiner(result_test)

        return Dataset(X_train=result_train, y_train=y, X_test=result_test)

    def find_weights(self, scorer, test_size=0.2, method='SLSQP'):
        """Finds optimal weights for weighted average of models.

        Parameters
        ----------
        scorer : function
            Scikit-learn like metric.
        test_size : float, default 0.2
        method : str
            Type of solver.  Should be one of:

            - 'Nelder-Mead'
            - 'Powell'
            - 'CG'
            - 'BFGS'
            - 'Newton-CG'
            - 'L-BFGS-B'
            - 'TNC'
            - 'COBYLA'
            - 'SLSQP'
            - 'dogleg'
            - 'trust-ncg'

        Returns
        -------
        list
        """
        p = Optimizer(self.models, test_size=test_size, scorer=scorer)
        return p.minimize(method)

    def weight(self, weights):
        """Applies weighted mean to models.

        Parameters
        ----------
        weights : list

        Returns
        -------
        np.ndarray

        Examples
        ----------
        >>> pipeline = ModelsPipeline(model_rf,model_lr)
        >>> pipeline.weight([0.8,0.2])
        """
        return self.apply(lambda x: np.average(x, axis=0, weights=weights))


class PipeApply(object):
    def __init__(self, function, models):
        self.models = models
        self.function = function

    def execute(self):
        results = []
        for model in self.models:
            results.append(model.predict())
        return self.function(results)

    def validate(self, scorer=None, k=1, test_size=0.1, stratify=False, shuffle=True, seed=100, indices=None):
        params = dict(k=k, test_size=test_size, stratify=stratify, scorer=scorer,
                      shuffle=shuffle, seed=seed, indices=indices)
        scores = []
        scorer = params['scorer']
        params['scorer'] = None
        y_preds_grouped, y_true_grouped = group_models(self.models, params)
        for i in y_preds_grouped.keys():
            result = self.function(y_preds_grouped[i])
            scores.append(scorer(y_true_grouped[i], result))
        report_score(scores, scorer)
        return scores
