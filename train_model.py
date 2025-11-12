import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
data = pd.read_csv("dataset/students.csv")

print(f"Dataset loaded: {len(data)} students")

# Features (input data)
X = data[["attendance", "assignment_score", "exam_score", "participation"]]

# Target (what we want to predict)
y = data["final_grade"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "student_model.pkl")
print("Model saved as 'student_model.pkl'")