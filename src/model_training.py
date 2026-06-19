"""Train PD models: Logistic Regression (scorecard-ready) and XGBoost (challenger)."""
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from . import config


def train_logistic(X_woe, y):
    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=config.RANDOM_STATE)
    model.fit(X_woe, y)
    return model


def train_xgboost(X, y):
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
