from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelValidator:
    def __init__(self, model, test_y, avoid_model = True, test_x = None, predicted_y = None):
        self.model = model

        self.test_y = test_y
        self.test_x = test_x

        self.model_results = None
        self.predicted_y = predicted_y

        self.avoid_model = avoid_model

        if avoid_model==True:
            self.model = None

        self.metrics = None


    def get_model(self):
        return self.model

    def get_model_results(self):
        if self.avoid_model==True:
            mse = mean_squared_error(self.test_y, self.predicted_y)
            rmse = pow(mse,0.5)
            mae = mean_absolute_error(self.test_y, self.predicted_y)
            r2_score = r2_score(self.test_y, self.predicted_y)
            self.metrics = dict()
            self.metrics['mse'] = mse
            self.metrics['rmse'] = rmse
            self.metrics['mae'] = mae
            self.metrics['r2_score'] = r2_score
        
        elif self.avoid_model!=True:
            self.metrics = dict()
            y_pred = self.model.predict(self.test_x)
            mse = mean_squared_error(self.test_y, y_pred)
            rmse = pow(mse,0.5)
            mae = mean_absolute_error(self.test_y, y_pred)
            r2_score = r2_score(self.test_y, y_pred)

            self.metrics['mse'] = mse
            self.metrics['rmse'] = rmse
            self.metrics['mae'] = mae
            self.metrics['r2_score'] = r2_score

    def get_metrics(self):
        return self.metrics