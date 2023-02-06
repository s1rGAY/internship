from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelValidator:
    def __init__(self, model, test_x, test_y):
        self.model = model
        self.test_y = test_y
        self.test_x = test_x
        self.model_results = None

    def get_model(self):
        return self.model

    def get_model_results(self):
        y_pred = self.model.predict(self.test_x)
        
        mse = mean_squared_error(self.dev_y, y_pred)
        rmse = pow(mse,0.5)
        mae = mean_absolute_error(self.dev_y, y_pred)
        r2_score = r2_score(self.dev_y, y_pred)

        return mse, rmse, mae, r2_score