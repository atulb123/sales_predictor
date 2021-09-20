import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV, ElasticNet
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Predict:
    def predict_outcome(self, model_type, values=[]):
        transform = pickle.load(open(os.getcwd()+"/models/transform.pickle", "rb"))
        transformed_values = transform.transform([values])
        if model_type.lower() == "lasso":
            model = pickle.load(open(os.getcwd()+"/models/lasso_model.pickle", "rb"))
        elif model_type.lower() == "elastic_cv":
            model = pickle.load(open(os.getcwd()+"/models/elastic_model.pickle", "rb"))
        else:
            model = pickle.load(open(os.getcwd()+"/models/linear_regression_model.pickle", "rb"))
        return str(model.predict(transformed_values)[0])
