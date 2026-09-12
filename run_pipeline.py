# Install once if required
# !pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap plotly streamlit joblib openpyxl

import os
import json
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
CLASS_DIR = OUTPUT_DIR / "classification"
REG_DIR = OUTPUT_DIR / "regression"
SHAP_DIR = OUTPUT_DIR / "shap"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, EDA_DIR, CLASS_DIR, REG_DIR, SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


DATASET_PATHS = [
    Path("LungCancer.txt"),
    Path("Pasted text.txt"),
    Path("/mnt/data/Pasted text.txt"),
    Path("/content/LungCancer.json")
]

dataset_path = next((p for p in DATASET_PATHS if p.exists()), None)

if dataset_path is None:
    raise FileNotFoundError(
        "Dataset not found. Place LungCancer.txt or LungCancer.json in the notebook folder."
    )

try:
    if dataset_path.suffix == ".json":
        df = pd.read_json(dataset_path)
    else:
        df = pd.read_csv(dataset_path, sep="\t")
except Exception:
    if dataset_path.suffix == ".json":
        df = pd.read_json(dataset_path)
    else:
        df = pd.read_csv(dataset_path, sep=None, engine="python")

df.columns = [str(c).strip() for c in df.columns]
df = df.replace([np.inf, -np.inf], np.nan)

print("Dataset:", dataset_path)
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

print(df.head())

csv_path = DATA_DIR / "lung_cancer.csv"
json_path = DATA_DIR / "lung_cancer.json"

df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", indent=2)

print("Created:", csv_path)
print("Created:", json_path)


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nData types:")
print(df.dtypes.to_frame("dtype"))

print("\nMissing values:")
print(df.isna().sum().to_frame("missing"))

print("\nDuplicate rows:", df.duplicated().sum())

if "Class" in df.columns:
    print("\nClass distribution:")
    print(df["Class"].value_counts().to_frame("count"))
    print((df["Class"].value_counts(normalize=True) * 100).round(2).to_frame("percentage"))


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols = [c for c in numeric_cols if c != "PatientID"]

stats_rows = []

for col in feature_cols:
    s = df[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    row = {
        "feature": col,
        "dtype": str(s.dtype),
        "count": int(s.count()),
        "missing_count": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean() * 100),
        "unique": int(s.nunique()),
        "zero_count": int((s == 0).sum()),
        "zero_pct": float((s == 0).mean() * 100),
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "variance": s.var(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": s.skew(),
        "kurtosis": s.kurt(),
        "coefficient_of_variation": (s.std()/s.mean()) if s.mean() != 0 else np.nan,
        "outlier_count_iqr": int(((s < lower) | (s > upper)).sum())
    }
    stats_rows.append(row)

feature_statistics = pd.DataFrame(stats_rows)
feature_statistics.to_csv(OUTPUT_DIR / "feature_statistics.csv", index=False)
print(feature_statistics)


if "Class" in df.columns:
    class_counts = df["Class"].value_counts()
    class_stats = pd.DataFrame({
        "count": class_counts,
        "percentage": (class_counts / len(df) * 100).round(2)
    })

    imbalance_ratio = class_counts.max() / class_counts.min()
    print("Class imbalance ratio:", round(imbalance_ratio, 3))
    print(class_stats)


if "Class" in df.columns:
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x="Class")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "class_distribution.png", dpi=200)
    # plt.show()


for col in feature_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution: {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{col}_distribution_boxplot.png", dpi=150)
    # plt.show()


if "Class" in df.columns:
    for col in feature_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, x="Class", y=col)
        plt.title(f"{col} by Class")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{col}_class_boxplot.png", dpi=150)
        # plt.show()


plt.figure(figsize=(20,16))
corr = df[feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=200)
# plt.show()


zero_pct = (df[feature_cols].eq(0).mean() * 100).sort_values(ascending=False)

plt.figure(figsize=(12,7))
zero_pct.plot(kind="bar")
plt.ylabel("Zero values (%)")
plt.title("Zero Percentage by Feature")
plt.tight_layout()
plt.savefig(EDA_DIR / "zero_percentage.png", dpi=180)
# plt.show()


skewness = df[feature_cols].skew().sort_values()

plt.figure(figsize=(12,7))
skewness.plot(kind="bar")
plt.ylabel("Skewness")
plt.title("Feature Skewness")
plt.tight_layout()
plt.savefig(EDA_DIR / "feature_skewness.png", dpi=180)
# plt.show()


quality_report = pd.DataFrame({
    "missing": df[feature_cols].isna().sum(),
    "missing_pct": df[feature_cols].isna().mean() * 100,
    "unique": df[feature_cols].nunique(),
    "zero_pct": df[feature_cols].eq(0).mean() * 100,
    "skewness": df[feature_cols].skew(),
    "variance": df[feature_cols].var()
})

print(quality_report)

print("Duplicate rows:", df.duplicated().sum())
print("Constant features:", [c for c in feature_cols if df[c].nunique() <= 1])


from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

X = df.drop(columns=["Class"], errors="ignore").copy()
X = X.drop(columns=["PatientID"], errors="ignore")
y = df["Class"].copy()

numeric_features = X.select_dtypes(include=np.number).columns.tolist()

missing_ratio = X[numeric_features].isna().mean().mean()

if missing_ratio > 0:
    imputer = KNNImputer(n_neighbors=5)
    print("Using KNNImputer because missing values are present.")
else:
    imputer = SimpleImputer(strategy="median")
    print("No missing values detected. Imputer remains in pipeline only for deployment robustness.")

preprocessor = Pipeline([
    ("imputer", imputer),
    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    ("scaler", RobustScaler())
])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

print(X_train.shape, X_test.shape)


from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    f_classif,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = X.columns.tolist()

variance_selector = VarianceThreshold(threshold=0.0)
X_var = variance_selector.fit_transform(X_train_processed)

kept_features = np.array(feature_names)[variance_selector.get_support()]
print("Features after VarianceThreshold:", len(kept_features))

mi = mutual_info_classif(X_train_processed, y_train, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({"feature": feature_names, "mutual_information": mi})
mi_df = mi_df.sort_values("mutual_information", ascending=False)
mi_df.to_csv(OUTPUT_DIR / "mutual_information_ranking.csv", index=False)
print(mi_df)


rf_selector_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
rf_selector_model.fit(X_train_processed, y_train)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_selector_model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)
print(importance_df)


pca = PCA()
pca.fit(X_train_processed)

pca_df = pd.DataFrame({
    "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
})

print(pca_df)
pca_df.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

plt.figure(figsize=(8,5))
plt.plot(pca_df["component"], pca_df["cumulative_variance"], marker="o")
plt.axhline(0.95, linestyle="--")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.tight_layout()
# plt.show()


from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(loss="log_loss", class_weight="balanced", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Gaussian NB": GaussianNB(),
    "Bernoulli NB": BernoulliNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Linear SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "RBF SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=2000, random_state=RANDOM_STATE)
}

try:
    from xgboost import XGBClassifier
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss"
    )
except Exception as e:
    print("XGBoost unavailable:", e)

try:
    from lightgbm import LGBMClassifier
    classifiers["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        verbose=-1
    )
except Exception as e:
    print("LightGBM unavailable:", e)

try:
    from catboost import CatBoostClassifier
    classifiers["CatBoost"] = CatBoostClassifier(
        iterations=300,
        random_state=RANDOM_STATE,
        verbose=0
    )
except Exception as e:
    print("CatBoost unavailable:", e)

print("Total classifiers:", len(classifiers))
print(list(classifiers.keys()))


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)

classification_results = []
trained_classifiers = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in classifiers.items():
    print("Training:", name)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    try:
        start = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        pred = pipe.predict(X_test)
        inference_time = time.time() - start

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "Precision Macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "Precision Weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall Macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "Recall Weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 Macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "F1 Weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
            "Cohen Kappa": cohen_kappa_score(y_test, pred),
            "Train Time": train_time,
            "Inference Time": inference_time
        }

        if hasattr(pipe, "predict_proba"):
            try:
                prob = pipe.predict_proba(X_test)
                result["ROC AUC OvR Macro"] = roc_auc_score(
                    y_test, prob, multi_class="ovr", average="macro"
                )
                result["ROC AUC OvR Weighted"] = roc_auc_score(
                    y_test, prob, multi_class="ovr", average="weighted"
                )
                result["Log Loss"] = log_loss(y_test, prob)
            except:
                result["ROC AUC OvR Macro"] = np.nan
                result["ROC AUC OvR Weighted"] = np.nan
                result["Log Loss"] = np.nan
        else:
            result["ROC AUC OvR Macro"] = np.nan
            result["ROC AUC OvR Weighted"] = np.nan
            result["Log Loss"] = np.nan

        classification_results.append(result)
        trained_classifiers[name] = pipe

    except Exception as e:
        print(f"{name} failed:", e)

classification_results_df = pd.DataFrame(classification_results)
classification_results_df = classification_results_df.sort_values(
    ["F1 Macro", "MCC", "Balanced Accuracy"],
    ascending=False
)

classification_results_df.to_csv(CLASS_DIR / "classification_model_comparison.csv", index=False)
print(classification_results_df)


best_classifier_name = classification_results_df.iloc[0]["Model"]
best_classifier = trained_classifiers[best_classifier_name]

print("Best classifier:", best_classifier_name)

best_pred = best_classifier.predict(X_test)

print(classification_report(
    y_test,
    best_pred,
    target_names=label_encoder.classes_
))

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_classifier_name}")
plt.tight_layout()
plt.savefig(CLASS_DIR / "best_classifier_confusion_matrix.png", dpi=200)
# plt.show()


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

if hasattr(best_classifier, "predict_proba"):
    probabilities = best_classifier.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

    plt.figure(figsize=(8,6))

    for i, class_name in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.3f}")

    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASS_DIR / "multiclass_roc_curve.png", dpi=200)
    # plt.show()


REGRESSION_TARGET = "CH2O"

if REGRESSION_TARGET not in df.columns:
    REGRESSION_TARGET = feature_cols[0]

X_reg = df.drop(columns=[REGRESSION_TARGET, "PatientID", "Class"], errors="ignore")
y_reg = df[REGRESSION_TARGET].astype(float)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Regression target:", REGRESSION_TARGET)
print("Predictors:", X_reg.shape[1])


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor
)
from sklearn.neural_network import MLPRegressor

regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "Huber": HuberRegressor(max_iter=2000),
    "SGD Regressor": SGDRegressor(max_iter=5000, random_state=RANDOM_STATE),
    "KNN Regressor": KNeighborsRegressor(),
    "Linear SVR": LinearSVR(max_iter=10000, random_state=RANDOM_STATE),
    "RBF SVR": SVR(kernel="rbf"),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Bagging": BaggingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostRegressor(n_estimators=150, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=3000, random_state=RANDOM_STATE)
}

try:
    from xgboost import XGBRegressor
    regressors["XGBoost"] = XGBRegressor(n_estimators=300, random_state=RANDOM_STATE)
except Exception as e:
    print("XGBoost unavailable:", e)

try:
    from lightgbm import LGBMRegressor
    regressors["LightGBM"] = LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
except Exception as e:
    print("LightGBM unavailable:", e)

try:
    from catboost import CatBoostRegressor
    regressors["CatBoost"] = CatBoostRegressor(iterations=300, random_state=RANDOM_STATE, verbose=0)
except Exception as e:
    print("CatBoost unavailable:", e)

print("Total regressors:", len(regressors))
print(list(regressors.keys()))


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error
)

reg_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    ("scaler", RobustScaler())
])

regression_results = []
trained_regressors = {}

def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - (1-r2)*(n-1)/(n-p-1)

for name, model in regressors.items():
    print("Training:", name)

    pipe = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model", model)
    ])

    try:
        start = time.time()
        pipe.fit(Xr_train, yr_train)
        train_time = time.time() - start

        start = time.time()
        pred = pipe.predict(Xr_test)
        inference_time = time.time() - start

        mse = mean_squared_error(yr_test, pred)
        r2 = r2_score(yr_test, pred)

        result = {
            "Model": name,
            "MAE": mean_absolute_error(yr_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2,
            "Adjusted R2": adjusted_r2(r2, len(yr_test), Xr_test.shape[1]),
            "Median Absolute Error": median_absolute_error(yr_test, pred),
            "Explained Variance": explained_variance_score(yr_test, pred),
            "Max Error": max_error(yr_test, pred),
            "Train Time": train_time,
            "Inference Time": inference_time
        }

        regression_results.append(result)
        trained_regressors[name] = pipe

    except Exception as e:
        print(f"{name} failed:", e)

regression_results_df = pd.DataFrame(regression_results)
regression_results_df = regression_results_df.sort_values(
    ["RMSE", "MAE"],
    ascending=True
)

regression_results_df.to_csv(REG_DIR / "regression_model_comparison.csv", index=False)
print(regression_results_df)


best_regressor_name = regression_results_df.iloc[0]["Model"]
best_regressor = trained_regressors[best_regressor_name]

reg_pred = best_regressor.predict(Xr_test)

print("Best regressor:", best_regressor_name)

plt.figure(figsize=(7,6))
plt.scatter(yr_test, reg_pred)
min_v = min(yr_test.min(), reg_pred.min())
max_v = max(yr_test.max(), reg_pred.max())
plt.plot([min_v,max_v], [min_v,max_v], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted — {best_regressor_name}")
plt.tight_layout()
plt.savefig(REG_DIR / "actual_vs_predicted.png", dpi=200)
# plt.show()

residuals = yr_test - reg_pred

plt.figure(figsize=(7,5))
plt.scatter(reg_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(REG_DIR / "residual_plot.png", dpi=200)
# plt.show()


try:
    import shap

    final_model = best_classifier.named_steps["model"]
    transformed_train = best_classifier.named_steps["preprocessor"].transform(X_train)
    transformed_test = best_classifier.named_steps["preprocessor"].transform(X_test)

    background = transformed_train[:min(100, len(transformed_train))]
    explain_data = transformed_test[:min(50, len(transformed_test))]

    if hasattr(final_model, "feature_importances_"):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(explain_data)

        shap.summary_plot(
            shap_values,
            explain_data,
            feature_names=X.columns.tolist(),
            show=False
        )
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "classification_shap_summary.png", dpi=200, bbox_inches="tight")
        # plt.show()
    else:
        predict_fn = (
            final_model.predict_proba
            if hasattr(final_model, "predict_proba")
            else final_model.predict
        )
        explainer = shap.Explainer(predict_fn, background)
        shap_values = explainer(explain_data)

        shap.plots.bar(shap_values, show=False)
        plt.savefig(SHAP_DIR / "classification_shap_bar.png", dpi=200, bbox_inches="tight")
        # plt.show()

except Exception as e:
    print("SHAP could not be completed automatically:", e)
    print("The selected model may require a model-specific SHAP explainer.")


artifacts = {
    "preprocessing_pipeline.pkl": preprocessor,
    "label_encoder.pkl": label_encoder,
    "best_classifier.pkl": best_classifier,
    "best_regressor.pkl": best_regressor,
    "selected_features.pkl": X.columns.tolist(),
    "classifier_metadata.pkl": {
        "model_name": best_classifier_name,
        "classes": label_encoder.classes_.tolist(),
        "features": X.columns.tolist()
    },
    "regressor_metadata.pkl": {
        "model_name": best_regressor_name,
        "target": REGRESSION_TARGET,
        "features": X_reg.columns.tolist()
    }
}

for filename, obj in artifacts.items():
    path = MODEL_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", path)


classification_metric_names = classification_results_df.columns.tolist()
regression_metric_names = regression_results_df.columns.tolist()

experiment_manifest = {
    "classification_models": list(classifiers.keys()),
    "classification_metrics": classification_metric_names,
    "regression_models": list(regressors.keys()),
    "regression_metrics": regression_metric_names,
    "classification_target": "Class",
    "regression_target": REGRESSION_TARGET
}

with open(OUTPUT_DIR / "experiment_manifest.json", "w") as f:
    json.dump(experiment_manifest, f, indent=2)

print(json.dumps(experiment_manifest, indent=2))


plt.figure(figsize=(12,7))
top = classification_results_df.head(15)
sns.barplot(data=top, y="Model", x="F1 Macro")
plt.title("Top Classification Models — F1 Macro")
plt.tight_layout()
plt.savefig(CLASS_DIR / "classification_f1_comparison.png", dpi=200)
# plt.show()

plt.figure(figsize=(12,7))
top_reg = regression_results_df.head(15)
sns.barplot(data=top_reg, y="Model", x="RMSE")
plt.title("Top Regression Models — RMSE")
plt.tight_layout()
plt.savefig(REG_DIR / "regression_rmse_comparison.png", dpi=200)
# plt.show()
