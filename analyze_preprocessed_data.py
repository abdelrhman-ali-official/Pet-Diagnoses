import pandas as pd

# Load the preprocessed dataset
preprocessed_dataset_path = "/home/ubuntu/pet_diagnosis_preprocessed.csv"
df = pd.read_csv(preprocessed_dataset_path)

output_analysis_path = "/home/ubuntu/preprocessed_data_analysis_report.txt"

report_content = "--- Preprocessed Dataset Analysis Report ---\n\n"
report_content += "--- Dataset Shape ---\n"
report_content += str(df.shape) + "\n"

report_content += "\n--- Value Counts for Key Categorical Columns ---\n"
key_categorical_cols = ["Animal_Type", "Age_Category", "Primary_Condition", "Severity", "Treatment_Category"]

for col in key_categorical_cols:
    if col in df.columns:
        report_content += f"\n-- Column: {col} --\n"
        report_content += df[col].value_counts(dropna=False).to_string() + "\n"
    else:
        report_content += f"\n-- Column: {col} (Not Found) --\n"

# --- Save analysis to a file ---
with open(output_analysis_path, "w") as f:
    f.write(report_content)

print(f"Preprocessed dataset analysis complete. Results saved to {output_analysis_path}")
print("\n--- Value Counts for Primary_Condition (first 20) ---")
if "Primary_Condition" in df.columns:
    print(df["Primary_Condition"].value_counts().head(20).to_string())
else:
    print("Primary_Condition column not found.")

