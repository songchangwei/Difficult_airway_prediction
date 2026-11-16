from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_model(name, categorical_columns, numerical_columns, seed):
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns),
        ]
    )

    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=2,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=seed,
        )
    elif name == "adaboost":
        estimator = AdaBoostClassifier(
            n_estimators=500,
            learning_rate=0.02,
            random_state=seed,
        )
    elif name == "svm_rbf":
        estimator = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            p=2,
        )
    elif name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required; install requirements.txt dependencies")
        estimator = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
