import os
import joblib
import pandas as pd

from hyperparameter_tuning import Tuning
from models_blending import Blending

X = pd.read_csv('datasets/Real estate.csv').drop(['Y house price of unit area', 'No'], axis=1)
y = pd.read_csv('datasets/Real estate.csv')['Y house price of unit area']


def stage_1():
    print("Stage 1:")
    tuning = Tuning(X, y)

    if os.path.exists('study.pkl') is not True:
        tuning.make_study()
    else:
        tuning.load_study()
        print("Study loaded!")

    try:
        tuning.optimize_study()
    except Exception:
        tuning.save_study()
        tuning.print_result()
    finally:
        tuning.save_study()
        tuning.print_result()


def stage_2():
    print("Stage 2:")
    blending = Blending(X, y)
    blending.models_predictions()

    if os.path.exists('final_study.pkl') is not True:
        blending.make_study()
    else:
        blending.load_study()
        print("Study loaded!")

    try:
        blending.optimize_study()
    except Exception:
        blending.save_study()
        blending.print_result()
    finally:
        blending.save_study()
        blending.print_result()

        print("Stage 3:")
        blending.final_validation()


if __name__ == '__main__':
    stage_1()
    stage_2()
