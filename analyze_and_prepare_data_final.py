import pandas as pd

# Load the augmented dataset
augmented_data_path = "/home/ubuntu/pet_diagnosis_augmented.csv"
df = pd.read_csv(augmented_data_path)

print(f"Original augmented dataset shape: {df.shape}")

# Calculate class distribution for the target variable 'Primary_Condition'
class_counts = df["Primary_Condition"].value_counts()

# Identify classes with fewer than 2 samples (minimum required for stratified split)
min_samples_threshold = 2
rare_classes = class_counts[class_counts < min_samples_threshold].index.tolist()

report_lines = []
report_lines.append("Class Distribution Analysis and Preparation Report:")
report_lines.append(f"Loaded augmented dataset from: {augmented_data_path}")
report_lines.append(f"Original shape: {df.shape}")
report_lines.append(f"Target column: Primary_Condition")
report_lines.append(f"Minimum samples per class for stratified splitting: {min_samples_threshold}")

if rare_classes:
    report_lines.append(f"\nFound {len(rare_classes)} classes with fewer than {min_samples_threshold} samples:")
    for cls in rare_classes:
        report_lines.append(f"  - {cls} (Count: {class_counts[cls]})")
    
    # Remove rows belonging to these rare classes
    df_filtered = df[~df["Primary_Condition"].isin(rare_classes)]
    report_lines.append(f"\nRemoved {len(rare_classes)} rare classes.")
    report_lines.append(f"Shape after removing rare classes: {df_filtered.shape}")
    num_removed_rows = df.shape[0] - df_filtered.shape[0]
    report_lines.append(f"Number of rows removed: {num_removed_rows}")
else:
    report_lines.append("\nNo classes found with fewer than {min_samples_threshold} samples. No rows removed.")
    df_filtered = df.copy()

# Save the further refined dataset
final_prepared_data_path = "/home/ubuntu/pet_diagnosis_augmented_prepared.csv"
df_filtered.to_csv(final_prepared_data_path, index=False)
report_lines.append(f"\nPrepared dataset saved to: {final_prepared_data_path}")

# Save the report
report_file_path = "/home/ubuntu/class_distribution_preparation_report.txt"
with open(report_file_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"Class distribution analysis and preparation complete.")
print(f"Report saved to: {report_file_path}")
print(f"Prepared dataset saved to: {final_prepared_data_path}")

print("\n--- Value counts for Primary_Condition in Prepared Data (Top 20) ---")
print(df_filtered["Primary_Condition"].value_counts().nlargest(20))

