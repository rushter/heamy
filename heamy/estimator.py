# coding:utf-8
import hashlib
import inspect
import logging
from inspect import getargspec

import numpy as np

from .cache import Cache, np_hash
from .dataset import Dataset
from .utils.main import concat, tsplit, reshape_1d
from .utils.main import report_score

REQUIRED_ARGS = set(['X_train', 'y_train', 'X_test', 'y_test'])
logger = logging.getLogger('heamy.estimator')


class BaseEstimator(object):
    problem = None

    def __init__(self, dataset, estimator=None, parameters=None, name=None, use_cache=True):
        """Base class for estimators.
        This class should not be used directly."""
        if estimator is not None:
            self._estimator = estimator
        elif hasattr(self.__class__, 'estimator'):
            self._estimator = self.estimator
        else:
            raise ValueError('Missing required estimator object.')

        if callable(dataset):
            self.dataset = dataset()
        else:
            self.dataset = dataset

        if (not use_cache) and (not self.dataset.loaded):
            self.dataset.load()

        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = {}

        self._hash = None
        self.probability = False
        self.use_cache = use_cache
        self._check_estimator()

        if name is None:
            name = '%s(%s)' % (self.estimator_name, self.hash)
        self._name = name

    def _check_estimator(self):
        est = self._estimator
        self._is_class = isinstance(est, type)
        if not self._is_class:
            args = set(getargspec(est).args)
            if 'self' in args:
                args.remove('self')
            if not REQUIRED_ARGS.issubset(args):
                raise ValueError('Missing required arguments. Please specify %s' % ','.join(REQUIRED_ARGS))

    @property
    def estimator_name(self):
        if hasattr(self.__class__, 'estimator'):
            name = self.__class__.__name__
        else:
            name = self._estimator.__name__
        return name

    @property
    def name(self):
        return self._name

    @property
    def hash(self):
        if self._hash is None:
            m = hashlib.new('md5')
            for key in sorted(self.parameters.keys()):
                h_string = ('%s-%s' % (key, self._convert_parameter(self.parameters[key]))).encode('utf-8')
                m.update(h_string)
            m.update(self.estimator_name.encode('utf-8'))
            m.update(self.dataset.hash.encode('utf-8'))

            if not self._is_class:
                m.update(inspect.getsource(self._estimator).encode('utf-8'))
            self._hash = m.hexdigest()

        return self._hash

    def __repr__(self):
        return self._name

    def _convert_parameter(self, x):
        """If callable then return object's source code."""
        if callable(x):
            return inspect.getsource(x)
        return x

    def _predict(self, X_train, y_train, X_test, y_test=None):
        if self._is_class:
            estimator = self._estimator(**self.parameters)
            estimator.fit(X_train, y_train)
            if self.probability:
                result = estimator.predict_proba(X_test)
            else:
                result = estimator.predict(X_test)

            if self.problem == 'classification' and self.probability:
                # Return second column for binary classification
                if len(result.shape) == 2 and result.shape[1] == 2:
                    result = result[:, 1]
        else:
            result = self._estimator(X_train=X_train, y_train=y_train,
                                     X_test=X_test, y_test=y_test, **self.parameters)

        return result

    def predict(self):
        if self.use_cache:
            c = Cache(self.hash, prefix='p')
            if c.available:
                logger.info('Loading %s\'s prediction from cache.' % self._name)
                prediction = c.retrieve('prediction')
                return prediction
            elif not self.dataset.loaded:
                self.dataset.load()

        prediction = self._predict(X_train=self.dataset.X_train, y_train=self.dataset.y_train,
                                   X_test=self.dataset.X_test)
        if self.use_cache:
            c.store('prediction', prediction)

        return prediction

    def _dhash(self, params):
        """Get hash of the dictionary object."""
        m = hashlib.new('md5')
        m.update(self.hash.encode('utf-8'))
        for key in sorted(params.keys()):
            h_string = ('%s-%s' % (key, params[key])).encode('utf-8')
            m.update(h_string)
        return m.hexdigest()

    def validate(self, scorer=None, k=1, test_size=0.1, stratify=False, shuffle=True, seed=100, indices=None):
        """Evaluate a score by cross-validation.

        Parameters
        ----------
        scorer : function(y_true,y_pred), default None
            Scikit-learn like metric that returns a score.
        k : int, default 1
            The number of folds for validation.

            If k=1 then randomly split X_train into two parts otherwise use K-fold approach.
        test_size : float, default 0.1
            Size of the test holdout if k=1.
        stratify : bool, default False
        shuffle : bool, default True
        seed : int, default 100
        indices : list(np.array,np.array), default None
            Two numpy arrays that contain indices for train/test slicing. (train_index,test_index)

        Returns
        -------
        y_true: list
            Actual labels.
        y_pred: list
            Predicted labels.

        Examples
        --------
        >>> # Custom indices
        >>> train_index = np.array(range(250))
        >>> test_index = np.array(range(250,333))
        >>> res = model_rf.validate(mean_absolute_error,indices=(train_index,test_index))
        """
        if self.use_cache:
            pdict = {'k': k, 'stratify': stratify, 'shuffle': shuffle, 'seed': seed, 'test_size': test_size}

            if indices is not None:
                pdict['train_index'] = np_hash(indices[0])
                pdict['test_index'] = np_hash(indices[1])

            dhash = self._dhash(pdict)
            c = Cache(dhash, prefix='v')
            if c.available:
                logger.info('Loading %s\'s validation results from cache.' % self._name)
            elif (self.dataset.X_train is None) and (self.dataset.y_train is None):
                self.dataset.load()

        scores = []
        y_true = []
        y_pred = []

        if k == 1:
            X_train, y_train, X_test, y_test = self.dataset.split(test_size=test_size, stratify=stratify,
                                                                  seed=seed, indices=indices)
            if self.use_cache and c.available:
                prediction = c.retrieve('0')
            else:
                prediction = self._predict(X_train, y_train, X_test, y_test)
                if self.use_cache:
                    c.store('0', prediction)
            if scorer is not None:
                scores.append(scorer(y_test, prediction))
            y_true.append(y_test)
            y_pred.append(prediction)

        else:
            for i, fold in enumerate(self.dataset.kfold(k, stratify=stratify, seed=seed, shuffle=shuffle)):
                X_train, y_train, X_test, y_test, train_index, test_index = fold
                if self.use_cache and c.available:
                    prediction = c.retrieve(str(i))
                else:
                    prediction = None

                if prediction is None:
                    logger.info('Calculating %s\'s fold #%s' % (self._name, i + 1))
                    prediction = self._predict(X_train, y_train, X_test, y_test)
                    if self.use_cache:
                        c.store(str(i), prediction)
                if scorer is not None:
                    scores.append(scorer(y_test, prediction))
                y_true.append(y_test)
                y_pred.append(prediction)

        if scorer is not None:
            report_score(scores, scorer)

        return y_true, y_pred

    def stack(self, k=5, stratify=False, shuffle=True, seed=100, full_test=True):
        """Stack a single model. You should rarely be using this method. Use `ModelsPipeline.stack` instead.

        Parameters
        ----------
        k : int, default 5
        stratify : bool, default False
        shuffle : bool, default True
        seed : int, default 100
        full_test : bool, default True
            If `True` then evaluate test dataset on the full data otherwise take the mean of every fold.

        Returns
        -------
        `Dataset` with out of fold predictions.
        """
        train = None
        test = []

        if self.use_cache:
            pdict = {'k': k, 'stratify': stratify, 'shuffle': shuffle, 'seed': seed, 'full_test': full_test}
            dhash = self._dhash(pdict)
            c = Cache(dhash, prefix='s')
            if c.available:
                logger.info('Loading %s\'s stack results from cache.' % self._name)
                train = c.retrieve('train')
                test = c.retrieve('test')
                y_train = c.retrieve('y_train')
                return Dataset(X_train=train, y_train=y_train, X_test=test)
            elif not self.dataset.loaded:
                self.dataset.load()

        for i, fold in enumerate(self.dataset.kfold(k, stratify=stratify, seed=seed, shuffle=shuffle)):
            X_train, y_train, X_test, y_test, train_index, test_index = fold
            logger.info('Calculating %s\'s fold #%s' % (self._name, i + 1))
            if full_test:
                prediction = reshape_1d(self._predict(X_train, y_train, X_test, y_test))
            else:
                xt_shape = X_test.shape[0]
                x_t = concat(X_test, self.dataset.X_test)
                prediction_concat = reshape_1d(self._predict(X_train, y_train, x_t))
                prediction, prediction_test = tsplit(prediction_concat, xt_shape)
                test.append(prediction_test)

            if train is None:
                train = np.zeros((self.dataset.X_train.shape[0], prediction.shape[1]))

            train[test_index] = prediction

        if full_test:
            logger.info('Calculating %s\'s test data' % self._name)
            test = self._predict(self.dataset.X_train, self.dataset.y_train, self.dataset.X_test)
        else:
            test = np.mean(test, axis=0)

        test = reshape_1d(test)

        if self.use_cache:
            c.store('train', train)
            c.store('test', test)
            c.store('y_train', self.dataset.y_train)

        return Dataset(X_train=train, y_train=self.dataset.y_train, X_test=test)

    def blend(self, proportion=0.2, stratify=False, seed=100, indices=None):
        """Blend a single model.
        You should rarely be using this method. Use `ModelsPipeline.blend` instead.

        Parameters
        ----------
        proportion : float, default 0.2
            Test size holdout.
        stratify : bool, default False
        seed : int, default 100
        indices : list(np.ndarray,np.ndarray), default None
            Two numpy arrays that contain indices for train/test slicing. (train_index,test_index)

        Returns
        -------
        `Dataset`
        """

        if self.use_cache:
            pdict = {'proportion': proportion, 'stratify': stratify, 'seed': seed, 'indices': indices}

            if indices is not None:
                pdict['train_index'] = np_hash(indices[0])
                pdict['test_index'] = np_hash(indices[1])

            dhash = self._dhash(pdict)
            c = Cache(dhash, prefix='b')
            if c.available:
                logger.info('Loading %s\'s blend results from cache.' % self._name)
                train = c.retrieve('train')
                test = c.retrieve('test')
                y_train = c.retrieve('y_train')
                return Dataset(X_train=train, y_train=y_train, X_test=test)
            elif not self.dataset.loaded:
                self.dataset.load()

        X_train, y_train, X_test, y_test = self.dataset.split(test_size=proportion, stratify=stratify,
                                                              seed=seed, indices=indices)

        xt_shape = X_test.shape[0]
        x_t = concat(X_test, self.dataset.X_test)
        prediction_concat = reshape_1d(self._predict(X_train, y_train, x_t))
        new_train, new_test = tsplit(prediction_concat, xt_shape)
        if self.use_cache:
            c.store('train', new_train)
            c.store('test', new_test)
            c.store('y_train', y_test)
        return Dataset(new_train, y_test, new_test)


class Regressor(BaseEstimator):
    """Wrapper for regression problems.

    Parameters
    ----------
    dataset : `Dataset` object
    estimator : a callable scikit-learn like interface, custom function/class, optional
    parameters : dict, optional
        Arguments for `estimator` object.
    name : str, optional
        The unique name of `Estimator` object.
    use_cache : bool, optional
        if `True` then validate/predict/stack/blend results will be cached."""
    problem = 'regression'


class Classifier(BaseEstimator):
    """Wrapper for classification problems.

    Parameters
    ----------
    dataset : `Dataset` object
    estimator : a callable scikit-learn like interface, custom function/class, optional
    parameters : dict, optional
        Arguments for `estimator` object.
    name : str, optional
        The unique name of `Estimator` object.
    use_cache : bool, optional
        if `True` then validate/predict/stack/blend results will be cached."""
    problem = 'classification'

    def __init__(self, dataset, estimator=None, parameters=None, name=None, use_cache=True, probability=True):
        super(Classifier, self).__init__(dataset=dataset, estimator=estimator, parameters=parameters, name=name,
                                         use_cache=use_cache)
        self.probability = probability
