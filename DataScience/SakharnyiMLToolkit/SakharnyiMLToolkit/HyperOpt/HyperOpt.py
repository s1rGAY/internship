from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_squared_error

class Model_optimization:
    def __init__(self, model, train_x, train_y, test_x, test_y):
        self.model = model
        self.params = dict()
        self.best_params = dict()

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


    def set_model_params(self, params):
        self.params = params

    def operate(self, params):
        self.model = self.model(**self.params)
    
        self.model.fit(self.train_x, self.train_y)
        y_pred = self.model.predict(self.test_x)
        mse = self.mean_squared_error(self.test_y, self.y_pred)
        return {'loss': mse, 'params': params, 'status': 'ok'}

    def tune_model(self, max_evals):
        trials = Trials()
        self.best_params = fmin(self.operate, self.params, algo=tpe.suggest, max_evals=max_evals, trials=trials)
