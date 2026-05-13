import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# LOAD DATA
# =====================================

file_path = "Khảo-sát-thói-quen-học-tập-của-sinh-viên-Câu-trả-lời.xlsx"

df = pd.read_excel(file_path)

# =====================================
# RENAME COLUMNS
# =====================================

df = df.rename(columns={

    "Mỗi tuần, bạn thường dành ra bao nhiêu giờ để học? - Chỉ ghi số":
        "study_hours",

    "Hiện tại bạn đang học bao nhiêu môn? - Chỉ ghi số":
        "subjects",

    "Mỗi ngày, bạn thường ngủ bao nhiêu giờ?":
        "sleep_hours",

    "Mỗi ngày, bạn thường sử dụng mạng xã hội bao lâu?":
        "social_media_hours",

    "Bạn có đang làm thêm không?":
        "part_time_job",

    "Bạn có đang tham gia bất kỳ CLB nào không?":
        "club",

    "Phần trăm tham gia các buổi học trên trường của bạn khoảng bao nhiêu?":
        "attendance",

    "Bạn hay thường học theo cách nào?":
        "study_type",

    "GPA hiện tại của bạn là bao nhiêu? (Theo thang 4.0)":
        "gpa"
})

# =====================================
# CLEAN DATA
# =====================================

def convert_range_to_number(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    # Ví dụ: 11-12
    if "-" in value:

        parts = value.split("-")

        numbers = []

        for p in parts:

            p = ''.join(
                c for c in p
                if c.isdigit() or c == '.'
            )

            if p != "":
                numbers.append(float(p))

        if len(numbers) > 0:
            return sum(numbers) / len(numbers)

    # Ví dụ: 5h, 90%
    cleaned = ''.join(
        c for c in value
        if c.isdigit() or c == '.'
    )

    if cleaned == "":
        return np.nan

    return float(cleaned)

# Numeric columns

numeric_columns = [
    "study_hours",
    "subjects",
    "sleep_hours",
    "social_media_hours",
    "attendance",
    "gpa"
]

for col in numeric_columns:

    df[col] = df[col].apply(
        convert_range_to_number
    )

# =====================================
# FEATURE ENGINEERING
# =====================================

df["study_efficiency"] = (
    df["study_hours"] /
    (df["social_media_hours"] + 1)
)

df["good_sleep"] = df[
    "sleep_hours"
].apply(
    lambda x: 1 if 6 <= x <= 8 else 0
)

# =====================================
# FEATURES
# =====================================

features = [

    "study_hours",

    "subjects",

    "sleep_hours",

    "social_media_hours",

    "part_time_job",

    "club",

    "attendance",

    "study_type",

    "study_efficiency",

    "good_sleep"
]

X = df[features]

y = df["gpa"]

# =====================================
# NUMERIC / CATEGORICAL
# =====================================

numeric_features = [

    "study_hours",

    "subjects",

    "sleep_hours",

    "social_media_hours",

    "attendance",

    "study_efficiency",

    "good_sleep"
]

categorical_features = [

    "part_time_job",

    "club",

    "study_type"
]

# =====================================
# PREPROCESSING
# =====================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =====================================
# MODELS
# =====================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
}

# =====================================
# SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

results = []

best_pipeline = None
best_r2 = -999

# =====================================
# TRAIN ALL MODELS
# =====================================

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train

    pipeline.fit(
        X_train,
        y_train
    )

    # Predict

    predictions = pipeline.predict(
        X_test
    )

    # Metrics

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Save results

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })

    # Print result

    print(f"\n===== {name} =====")

    print(f"MAE  : {mae:.3f}")

    print(f"RMSE : {rmse:.3f}")

    print(f"R²   : {r2:.3f}")

    # Save best model

    if r2 > best_r2:

        best_r2 = r2

        best_pipeline = pipeline

# =====================================
# CREATE MODELS FOLDER
# =====================================

os.makedirs(
    "models",
    exist_ok=True
)

# =====================================
# SAVE BEST MODEL
# =====================================

joblib.dump(
    best_pipeline,
    "models/gpa_model.pkl"
)

print("\nBest model saved successfully!")

# =====================================
# FEATURE IMPORTANCE
# =====================================

try:

    importance = best_pipeline.named_steps[
        "model"
    ].feature_importances_

    feature_names = best_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n===== FEATURE IMPORTANCE =====")

    print(importance_df)

except:

    print(
        "\nFeature importance not supported for this model."
    )