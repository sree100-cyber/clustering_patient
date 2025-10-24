# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- 1. Load your dataset ---
df = pd.read_csv('your_dataset.csv')  # replace with your CSV path

# --- 2. Prepare features and target ---
X = df.drop(columns=['heart_disease'])
y = df['heart_disease']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Save model and scaler using pickle ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
