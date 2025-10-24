import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier  # Replace with your classifier
from sklearn.metrics import accuracy_score

# --- Load dataset ---
df = pd.read_csv("patient_dataset_with_clusters.csv")

# --- Define target and features ---
TARGET = "heart_disease"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- Identify categorical columns for encoding ---
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- One-hot encode categorical columns ---
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# --- Save feature columns for Streamlit ---
joblib.dump(X_encoded.columns, "feature_columns.pkl")

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# --- Scale numeric features ---
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- Save scaler for Streamlit ---
joblib.dump(scaler, "scaler.pkl")

# --- Train model ---
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# --- Save trained model ---
joblib.dump(model, "heart_disease_model.pkl")
print("Model, scaler, and feature columns saved successfully!")
