import optuna
import numpy as np
import pandas as pd
import joblib
import os
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold


class Tuning:

    def __init__(self, X=None, y=None):
        self.study = None
        self.previous_best_value = 10e6
        self.X = X
        self.y = y

    def objective(self, trial):
        params = {
            'metric': 'mae',
            'device': 'gpu',
            "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt', 'goss', 'dart']),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5),
            "max_depth": trial.suggest_int("max_depth", 1, 10000),
            'num_leaves': trial.suggest_int('num_leaves', 1, 10000),
            'n_estimators': trial.suggest_int('n_estimators', 1, 10000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
        }

        model = lgb.LGBMRegressor(**params)

        result = np.mean(
            np.abs(cross_val_score(model, self.X, self.y, cv=10, scoring='neg_mean_absolute_percentage_error')))
        return result

    def make_study(self):
        self.study = optuna.create_study(direction='minimize')

    def optimize_study(self):
        self.study.optimize(self.objective, timeout=600)
        if self.previous_best_value != self.study.best_value:
            self.save_study()
            self.make_oof_models()
        else:
            self.save_study()

    def save_study(self):
        joblib.dump(self.study, 'study.pkl')

    def load_study(self):
        self.study = joblib.load('study.pkl')
        self.previous_best_value = self.study.best_value

    def print_result(self):
        print(f"Best value: \n\t{self.study.best_value}")
        print(
            f"Result increased on: \n\t{round((np.abs(self.study.best_value - self.previous_best_value) * 100) / self.previous_best_value, 5)}%")
        print(f"Best params: \n\t{self.study.best_params}")

    def make_oof_models(self):
        df = pd.DataFrame(self.X).merge(pd.Series(self.y, name='target'), left_index=True, right_index=True)
        kf = KFold(n_splits=10, shuffle=True, random_state=2021)
        kf.get_n_splits(df)
        if os.path.exists('models') is not True:
            os.mkdir(os.path.join(os.getcwd(), 'models'))

        counter = 0
        for train_index, test_index in kf.split(df):
            model = lgb.LGBMRegressor(**self.study.best_params).fit(df.drop(['target'], axis=1).iloc[train_index],
                                                                    df.iloc[train_index]['target'])
            joblib.dump(model, 'models/model_' + str(counter) + '.pkl')
            counter += 1
