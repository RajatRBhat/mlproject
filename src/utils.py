import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as fobj:
            pickle.dump(obj, fobj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            logging.info(f"performing tuing on {model_name}")
            gscv = GridSearchCV(model, param_grid=params, cv=3)
            gscv.fit(X_train, y_train)

            logging.info(f"Best score for {model_name}: {gscv.best_score_}")
            logging.info(f"Best params for {model_name}: {gscv.best_params_}")

            model.set_params(**gscv.best_params_)
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

