# coding:utf-8
from scipy.optimize import minimize


class Optimizer(object):
    def __init__(self, models, scorer, test_size=0.2):
        self.test_size = test_size
        self.scorer = scorer
        self.models = models
        self.predictions = []
        self.y = None

        self._predict()

    def _predict(self):
        for model in self.models:
            y_true_list, y_pred_list = model.validate(k=1, test_size=self.test_size)
            if self.y is None:
                self.y = y_true_list[0]
            self.predictions.append(y_pred_list[0])

    def loss_func(self, weights):
        final_prediction = 0
        for weight, prediction in zip(weights, self.predictions):
            final_prediction += weight * prediction
        return self.scorer(self.y, final_prediction)

    def minimize(self, method):
        starting_values = [0.5] * len(self.predictions)
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * len(self.predictions)
        res = minimize(self.loss_func, starting_values, method=method, bounds=bounds, constraints=cons)
        print('Best Score (%s): %s' % (self.scorer.__name__, res['fun']))
        print('Best Weights: %s' % res['x'])
        return res['x']
