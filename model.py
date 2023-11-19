import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("preprocessed_data.csv")

    train, test = train_test_split(data)

    train_x = train.drop(["charges"], axis=1)
    test_x = test.drop(["charges"], axis=1)
    train_y = train[["charges"]]
    test_y = test[["charges"]]

    # Define a list of regression algorithms and their corresponding hyperparameters
    regressors = [
        {"name": "ElasticNet", "model": ElasticNet(), "params": {"alpha": 0.5, "l1_ratio": 0.5}},
        {"name": "LinearRegression", "model": LinearRegression(), "params": {}},
        {"name": "Ridge", "model": Ridge(), "params": {"alpha": 1.0}},
        {"name": "Lasso", "model": Lasso(), "params": {"alpha": 1.0}},
        {"name": "RandomForest", "model": RandomForestRegressor(), "params": {"n_estimators": 100, "random_state": 42}},
        {"name": "DecisionTree", "model": DecisionTreeRegressor(), "params": {"random_state": 42}},
        {"name": "SVR", "model": SVR(), "params": {"kernel": "linear"}},
        {"name": "KNeighbors", "model": KNeighborsRegressor(), "params": {"n_neighbors": 5}},
        {"name": "AdaBoost", "model": AdaBoostRegressor(), "params": {"n_estimators": 50, "random_state": 42}},
        {"name": "XGBoost", "model": XGBRegressor(), "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}},
        {"name": "GradientBoosting", "model": GradientBoostingRegressor(), "params": {"n_estimators": 100, "random_state": 42}},
    ]

    for regressor in regressors:
        with mlflow.start_run(run_name=regressor["name"]):
            model_name = regressor["name"]
            model = regressor["model"]
            params = regressor["params"]

            # Set hyperparameters if provided
            model.set_params(**params)

            model.fit(train_x, train_y)
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print(f"{model_name} model:")
            print("  RMSE:", rmse)
            print("  MAE:", mae)
            print("  R2:", r2)

            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            
            

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name=f"{model_name}_Prediction")
            else:
                mlflow.sklearn.log_model(model, "model")
