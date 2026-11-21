import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Dataset location
DATASET_PATH = os.path.join("dataset", "disaster_data.csv")

# Model storage location
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "disaster_risk_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "severity_encoder.pkl")


# Features matching the prepared dataset
FEATURES = [
    "Human_fatality",
    "Human_injured",
    "Population",
    "Mean_Flood_Durationt",
    "Corrected_Percent_Flooded_Area"
]

TARGET = "severity"


def train_disaster_model():

    # Check dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found! Run prepare_training_data.py first."
        )

    # Load dataset
    data = pd.read_csv(DATASET_PATH)

    # Check required columns
    required_columns = FEATURES + [TARGET]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing column: {column}")

    # Remove incomplete rows
    data = data.dropna(subset=required_columns)

    if len(data) < 10:
        raise ValueError("At least 10 complete records are required.")

    # Prepare features and target
    X = data[FEATURES]
    y = data[TARGET].astype(str)

    if y.nunique() < 2:
        raise ValueError("At least two severity classes are required.")

    # Encode severity labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n================================")
    print("DISASTER MODEL TRAINING COMPLETE")
    print("================================")

    print("Total records:", len(data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("\nModel saved successfully!")
    print("Model file:", MODEL_PATH)
    print("Encoder file:", ENCODER_PATH)


def predict_disaster_risk(
    human_fatality,
    human_injured,
    population,
    mean_flood_duration,
    corrected_percent_flooded_area
):

    # Check model files
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not trained yet. Run disaster_model.py first."
        )

    # Load trained model
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Prepare input
    input_data = pd.DataFrame([{
        "Human_fatality": human_fatality,
        "Human_injured": human_injured,
        "Population": population,
        "Mean_Flood_Durationt": mean_flood_duration,
        "Corrected_Percent_Flooded_Area": corrected_percent_flooded_area
    }])

    # Predict severity
    prediction = model.predict(input_data)[0]

    severity = encoder.inverse_transform([prediction])[0]

    # Calculate prediction confidence
    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    return {
        "predicted_severity": severity,
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":
    train_disaster_model()