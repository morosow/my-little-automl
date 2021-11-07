import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

MODELS_PATH = os.path.join(os.getcwd(), 'models')


class Blending:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.study = None
        self.previous_best_value = 10e6
        self.models = []
        self.df_predictions = pd.DataFrame()
        self.read_models()

    def read_models(self):
        for idx, filename in enumerate(os.listdir(MODELS_PATH)):
            self.models.append(joblib.load(os.path.join(MODELS_PATH, filename)))

    def models_predictions(self):
        for idx, model in enumerate(self.models):
            self.df_predictions = self.df_predictions.merge(pd.Series(model.predict(self.X), name=str(idx)),
                                                            left_index=True, right_index=True, how='right')

    def average_predictions_error(self):
        return np.abs(mean_absolute_percentage_error(self.y, self.df_predictions.apply(np.mean, axis=1)))

    def objective_final(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 10000),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "poisson"]),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0, 0.5),
            "max_depth": trial.suggest_int("max_depth", 1, 10000),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 10000),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0001, 1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0001, 1),
        }

        model = RandomForestRegressor(**params)

        result = np.mean(
            np.abs(cross_val_score(model, self.df_predictions, self.y, cv=20, scoring='neg_mean_absolute_percentage_error')))
        return result

    def make_study(self):
        self.study = optuna.create_study(direction='minimize')

    def optimize_study(self):
        self.study.optimize(self.objective_final, timeout=600)
        self.save_study()

    def save_study(self):
        joblib.dump(self.study, 'final_study.pkl')

    def load_study(self):
        self.study = joblib.load('final_study.pkl')
        self.previous_best_value = self.study.best_value

    def print_result(self):
        print(f"Best value: \n\t{self.study.best_value}")
        print(
            f"Result increased on: \n\t{round((np.abs(self.study.best_value - self.previous_best_value) * 100) / self.previous_best_value, 5)} % ")
        print(f"Best params: \n\t{self.study.best_params}")

    def final_validation(self):
        final_model = RandomForestRegressor(**self.study.best_params)
        final_score = (
            cross_val_score(final_model, self.df_predictions, self.y, verbose=3, n_jobs=-1,
                            scoring='neg_mean_absolute_percentage_error', cv=20))
        print(f"MEAN: \n\t{np.mean(np.abs(final_score))}")
        print(f"MEDIAN: \n\t{np.median(np.abs(final_score))}")
        print(f"MAX: \n\t{np.max(np.abs(final_score))}")








