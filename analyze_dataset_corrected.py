import pandas as pd

# Load the dataset
dataset_path = "/home/ubuntu/upload/pet_diagnosis.csv"
df = pd.read_csv(dataset_path)

output_report_path = "/home/ubuntu/dataset_analysis_report.txt"

# --- 1. Initial Overview ---
report_content = "--- Dataset Analysis Report ---\n\n"
report_content += "--- Dataset Shape ---\n"
report_content += str(df.shape) + "\n"
report_content += "\n--- First 5 Rows ---\n"
report_content += df.head().to_string() + "\n"
report_content += "\n--- Data Types ---\n"
report_content += df.dtypes.to_string() + "\n"

# --- 2. Missing Values ---
report_content += "\n--- Missing Values per Column ---\n"
report_content += df.isnull().sum().to_string() + "\n"

# --- 3. Unique Values in Categorical Columns & Consistency Check ---
report_content += "\n--- Unique Values & Counts for Categorical Columns ---\n"
categorical_cols = ['Animal_Type', 'Age_Category', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Primary_Condition', 'Severity', 'Treatment_Category']
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in existing_categorical_cols:
    report_content += f"\n-- Column: {col} --\n"
    report_content += df[col].value_counts(dropna=False).to_string() + "\n"

# --- 4. Summary Statistics for Numerical Columns ---
report_content += "\n--- Summary Statistics for Numerical Columns ---\n"
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
if numerical_cols:
    report_content += df[numerical_cols].describe().to_string() + "\n"
else:
    report_content += "No numerical columns found for summary statistics.\n"

# --- Save analysis to a file ---
with open(output_report_path, "w") as f:
    f.write(report_content)

print(f"Dataset analysis complete. Results saved to {output_report_path}")

