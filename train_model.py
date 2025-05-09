import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

# Start timer
start_time = time.time()

# Load the augmented dataset
data_path = "/home/ubuntu/pet_diagnosis_augmented_prepared.csv"
df = pd.read_csv(data_path)

# Drop rows where Primary_Condition is NaN, if any, as it's the target
df.dropna(subset=["Primary_Condition"], inplace=True)

# Define features (X) and target (y)
X = df.drop("Primary_Condition", axis=1)
y = df["Primary_Condition"]

# Identify categorical and text features
text_features = ["All_Symptoms"]
categorical_features = ["Animal_Type", "Age_Category", "Severity", "Treatment_Category"]

# Create preprocessing pipelines for text and categorical features
text_transformer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features[0]), # TF-IDF for 'All_Symptoms'
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Define the model pipeline
# Using RandomForestClassifier as an initial model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Define parameter grid for GridSearchCV
# Reduced grid for faster initial training, can be expanded for more thorough optimization
param_grid = {
    'classifier__n_estimators': [100, 200], # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],    # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV
# Using cv=3 for faster initial run, can be increased (e.g., 5 or 10)
# n_jobs=-1 uses all available cores
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

print("Starting model training and hyperparameter tuning...")
# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\n--- Model Evaluation ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\n--- Classification Report on Test Set ---")
# Get all unique labels from both y_test and y_pred_test to handle cases where some classes might not be predicted
labels = sorted(list(set(y_test) | set(y_pred_test)))
report = classification_report(y_test, y_pred_test, zero_division=0, labels=labels)
print(report)

# Save the best model, preprocessor, and class labels
model_save_path = "/home/ubuntu/pet_diagnosis_model.joblib"
preprocessor_save_path = "/home/ubuntu/pet_diagnosis_preprocessor.joblib"
labels_save_path = "/home/ubuntu/pet_diagnosis_labels.joblib"

joblib.dump(best_model, model_save_path)
joblib.dump(preprocessor, preprocessor_save_path) # Save the fitted preprocessor from the best_model's pipeline
joblib.dump(y.unique().tolist(), labels_save_path) # Save all unique class labels from the original target variable

end_time = time.time()
training_duration = end_time - start_time

# Create a training report
report_content = f"""
Model Training and Optimization Report:

Dataset used: {data_path}
Dataset shape after loading and cleaning NaNs in target: {df.shape}
Training data shape: {X_train.shape}
Test data shape: {X_test.shape}

Model: RandomForestClassifier (within a scikit-learn Pipeline)
Preprocessing:
  - Text Features ('All_Symptoms'): TfidfVectorizer (max_features=5000, stop_words='english', ngram_range=(1,2))
  - Categorical Features ({', '.join(categorical_features)}): OneHotEncoder (handle_unknown='ignore')

Hyperparameter Tuning: GridSearchCV
  - Cross-validation folds (cv): 3
  - Scoring metric: accuracy
  - Parameter Grid Searched:
    'classifier__n_estimators': [100, 200]
    'classifier__max_depth': [None, 10, 20]
    'classifier__min_samples_split': [2, 5]
    'classifier__min_samples_leaf': [1, 2]

Best Parameters Found:
{grid_search.best_params_}

Performance:
  - Training Accuracy: {train_accuracy:.4f}
  - Test Accuracy: {test_accuracy:.4f}

Classification Report (Test Set):
{report}

Saved Artifacts:
  - Trained Model: {model_save_path}
  - Preprocessor: {preprocessor_save_path} (Note: This is the preprocessor from the best_model pipeline)
  - Class Labels: {labels_save_path}

Total training and optimization duration: {training_duration:.2f} seconds.
"""

report_file_path = "/home/ubuntu/model_training_report.txt"
with open(report_file_path, "w") as f:
    f.write(report_content)

print(f"\nModel training complete. Trained model and report saved.")
print(f"Model saved to: {model_save_path}")
print(f"Preprocessor saved to: {preprocessor_save_path}")
print(f"Class labels saved to: {labels_save_path}")
print(f"Training report saved to: {report_file_path}")


