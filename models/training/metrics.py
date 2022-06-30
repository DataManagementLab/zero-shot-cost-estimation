import copy

import numpy as np
from sklearn.metrics import mean_squared_error


class Metric:
    """
    Abstract class defining a metric used to evaluate the zero-shot cost model performance (e.g., Q-error)
    """

    def __init__(self, metric_prefix='val_', metric_name='metric', maximize=True, early_stopping_metric=False):
        self.maximize = maximize
        self.default_value = -np.inf
        if not self.maximize:
            self.default_value = np.inf
        self.best_seen_value = self.default_value
        self.last_seen_value = self.default_value
        self.metric_name = metric_prefix + metric_name
        self.best_model = None
        self.early_stopping_metric = early_stopping_metric

    def evaluate(self, model=None, metrics_dict=None, **kwargs):
        metric = self.default_value
        try:
            metric = self.evaluate_metric(**kwargs)
        except ValueError as e:
            print(f"Observed ValueError in metrics calculation {e}")
        self.last_seen_value = metric

        metrics_dict[self.metric_name] = metric
        print(f"{self.metric_name}: {metric:.4f} [best: {self.best_seen_value:.4f}]")

        best_seen = False
        if (self.maximize and metric > self.best_seen_value) or (not self.maximize and metric < self.best_seen_value):
            self.best_seen_value = metric
            best_seen = True
            if model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
        return best_seen


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mape', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape

    def evaluate_metric(self, labels=None, preds=None):
        raise NotImplementedError


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mse', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mape', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape


class QError(Metric):
    def __init__(self, percentile=50, min_val=0.1, **kwargs):
        super().__init__(metric_name=f'median_q_error_{percentile}', maximize=False, **kwargs)
        self.percentile = percentile
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        if not np.all(labels >= self.min_val):
            print("WARNING: some labels are smaller than min_val")
        preds = np.abs(preds)
        # preds = np.clip(preds, self.min_val, np.inf)

        q_errors = np.maximum(labels / preds, preds / labels)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
        median_q = np.percentile(q_errors, self.percentile)
        return median_q
