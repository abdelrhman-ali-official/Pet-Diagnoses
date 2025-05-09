import pandas as pd

# Load the dataset
dataset_path = "/home/ubuntu/upload/pet_diagnosis.csv"
df = pd.read_csv(dataset_path)

# --- 1. Combine Symptom Columns ---
# Create a new column 'All_Symptoms' by combining Symptom_1 to Symptom_5
# We will join the symptom strings, handling potential NaN values by converting them to empty strings first
symptom_cols = [f'Symptom_{i}' for i in range(1, 6)]

# Ensure all symptom columns exist before trying to access them
existing_symptom_cols = [col for col in symptom_cols if col in df.columns]

if existing_symptom_cols:
    df['All_Symptoms'] = df[existing_symptom_cols].fillna('').astype(str).agg(' '.join, axis=1)
    # Clean up extra spaces that might result from joining empty strings
    df['All_Symptoms'] = df['All_Symptoms'].str.replace(r'\s+', ' ', regex=True).str.strip()
else:
    print("Warning: No symptom columns found to combine.")
    df['All_Symptoms'] = '' # Create an empty column if no symptom columns are present

# --- 2. Drop Original Symptom Columns and 'Accuracy_Score' ---
columns_to_drop = existing_symptom_cols + ['Accuracy_Score']
# Only drop columns that actually exist in the dataframe
columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop_existing, inplace=True, errors='ignore')

# --- 3. Save the preprocessed dataset ---
preprocessed_dataset_path = "/home/ubuntu/pet_diagnosis_preprocessed.csv"
df.to_csv(preprocessed_dataset_path, index=False)

print(f"Preprocessing complete. Cleaned dataset saved to {preprocessed_dataset_path}")
print("\n--- First 5 rows of preprocessed data ---")
print(df.head().to_string())

