import pandas as pd
import re

def categorize_age(age_str):
    if pd.isna(age_str):
        return "Unknown"
    age_str = str(age_str).lower()
    if "month" in age_str:
        try:
            months = int(re.findall(r"\d+", age_str)[0])
            if months <= 6:
                return "Kitten/Puppy" # Simplified for now
            elif months <= 12:
                return "Juvenile"
            else:
                return "Adult"
        except:
            return "Unknown"
    elif "year" in age_str:
        try:
            years = int(re.findall(r"\d+", age_str)[0])
            if years < 1:
                return "Juvenile"
            elif years <= 7:
                return "Adult"
            else:
                return "Senior"
        except:
            return "Unknown"
    else:
        try:
            # Assuming age might be just a number (treat as years)
            age_num = int(age_str)
            if age_num < 1:
                return "Juvenile"
            elif age_num <= 7:
                return "Adult"
            else:
                return "Senior"
        except:
            return "Unknown"

# Load the original preprocessed dataset
original_df_path = "/home/ubuntu/pet_diagnosis_preprocessed.csv"
original_df = pd.read_csv(original_df_path)

# Load the Kaggle dataset
kaggle_df_path = "/home/ubuntu/pet_diagnosis_project/data/cleaned_animal_disease_prediction.csv"
kaggle_df = pd.read_csv(kaggle_df_path)

# Filter Kaggle data for Dogs and Cats
kaggle_df_pets = kaggle_df[kaggle_df["Animal_Type"].isin(["Dog", "Cat"])].copy()

# --- Data Transformation for Kaggle Dataset ---

# 1. Map Animal_Type (already compatible)

# 2. Transform Age to Age_Category
# First, let's see the unique values in original Age_Category to align
# Original categories: Adult, Senior, Puppy, Kitten
# We'll refine categorize_age to match these better
def categorize_age_refined(animal_type, age_str):
    if pd.isna(age_str):
        return "Unknown"
    age_str = str(age_str).lower()
    
    years = None
    months = None

    if "year" in age_str:
        try:
            years = int(re.findall(r"\d+", age_str)[0])
        except:
            pass # Could be like "<1 year"
    if "month" in age_str:
        try:
            months = int(re.findall(r"\d+", age_str)[0])
        except:
            pass
    
    if years is None and months is None:
        try:
            # If only a number, assume years for adult animals, months for young ones if context allows
            # This is tricky without more context, defaulting to years if ambiguous
            num = int(age_str)
            if num <= 2 and animal_type == "Dog": # Heuristic: numbers <=2 for dogs might be years or months
                 # if num <=1 then puppy, else adult
                 if num <=1: years = num 
                 else: years = num
            elif num <=2 and animal_type == "Cat":
                 if num <=1: years = num
                 else: years = num
            else:
                years = num
        except:
            return "Unknown"

    if years is not None:
        if animal_type == "Dog":
            if years < 1:
                return "Puppy"
            elif years <= 7:
                return "Adult"
            else:
                return "Senior"
        elif animal_type == "Cat":
            if years < 1:
                return "Kitten"
            elif years <= 7:
                return "Adult"
            else:
                return "Senior"
    
    if months is not None:
        if animal_type == "Dog":
            if months <= 12:
                return "Puppy"
            else: # > 12 months
                return "Adult"
        elif animal_type == "Cat":
            if months <= 12:
                return "Kitten"
            else:
                return "Adult"
                
    return "Adult" # Default if logic doesn't catch

kaggle_df_pets.loc[:, "Age_Category"] = kaggle_df_pets.apply(lambda row: categorize_age_refined(row["Animal_Type"], row["Age"]), axis=1)

# 3. Map Disease_Prediction to Primary_Condition
kaggle_df_pets.rename(columns={"Disease_Prediction": "Primary_Condition"}, inplace=True)

# 4. Create All_Symptoms string
symptom_cols_kaggle = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
boolean_symptom_cols = ["Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", 
                        "Labored_Breathing", "Lameness", "Skin_Lesions", 
                        "Nasal_Discharge", "Eye_Discharge"]

def combine_symptoms_kaggle(row):
    symptoms = []
    for col in symptom_cols_kaggle:
        if pd.notna(row[col]) and row[col].strip() != "":
            symptoms.append(str(row[col]).strip())
    for col in boolean_symptom_cols:
        if row[col] == "Yes" or row[col] == True:
            # Convert column name to a more readable symptom
            symptom_name = col.replace("_", " ").lower()
            symptoms.append(symptom_name)
    return ", ".join(list(set(symptoms))) # Use set to avoid duplicates from Symptom_X and boolean flags

kaggle_df_pets.loc[:, "All_Symptoms"] = kaggle_df_pets.apply(combine_symptoms_kaggle, axis=1)

# 5. Handle Severity and Treatment_Category (not present in Kaggle dataset)
# We will add these columns with a placeholder like "Unknown" or "Not_Specified"
kaggle_df_pets.loc[:, "Severity"] = "Unknown"
kaggle_df_pets.loc[:, "Treatment_Category"] = "Unknown"

# Select and reorder columns in Kaggle data to match original preprocessed data
final_kaggle_cols = ["Animal_Type", "Age_Category", "Primary_Condition", "Severity", "Treatment_Category", "All_Symptoms"]
augmented_df_from_kaggle = kaggle_df_pets[final_kaggle_cols]

# Concatenate the original preprocessed data with the transformed Kaggle data
combined_df = pd.concat([original_df, augmented_df_from_kaggle], ignore_index=True)

# Save the augmented dataset
augmented_output_path = "/home/ubuntu/pet_diagnosis_augmented.csv"
combined_df.to_csv(augmented_output_path, index=False)

# Save a report of the augmentation
report_content = f"""
Data Augmentation Report:

Original dataset path: {original_df_path}
Original dataset shape: {original_df.shape}

Kaggle dataset path: {kaggle_df_path}
Kaggle dataset (Dogs and Cats only) shape: {kaggle_df_pets.shape}

Transformed Kaggle data for augmentation shape: {augmented_df_from_kaggle.shape}

Augmented (combined) dataset path: {augmented_output_path}
Augmented (combined) dataset shape: {combined_df.shape}

Columns in augmented dataset: {list(combined_df.columns)}

Age Categorization Logic for Kaggle Data:
- Parsed 'Age' column (e.g., '2 months', '5 years').
- Mapped to 'Puppy', 'Kitten', 'Adult', 'Senior' based on Animal_Type and age.
  - Dog: <1 year = Puppy, 1-7 years = Adult, >7 years = Senior.
  - Cat: <1 year = Kitten, 1-7 years = Adult, >7 years = Senior.
  - Months were converted to years for categorization.

Symptom Combination Logic for Kaggle Data:
- Combined Symptom_1, Symptom_2, Symptom_3, Symptom_4.
- Incorporated boolean symptom flags (e.g., Vomiting, Diarrhea) as text symptoms.
- Created a single 'All_Symptoms' string, similar to the original dataset.

Missing Columns Handling (Severity, Treatment_Category):
- These columns were not present in the Kaggle dataset.
- They have been added to the augmented portion with the value 'Unknown'.

Source of Augmentation Data:
Kaggle Dataset: animal disease prediction
URL: https://www.kaggle.com/datasets/shijo96john/animal-disease-prediction
File used: cleaned_animal_disease_prediction.csv
"""

report_file_path = "/home/ubuntu/data_augmentation_report.txt"
with open(report_file_path, "w") as f:
    f.write(report_content)

print(f"Data augmentation complete. Augmented dataset saved to {augmented_output_path}")
print(f"Augmentation report saved to {report_file_path}")

# Display head of the augmented data for verification
print("\n--- First 5 rows of Augmented Data ---")
print(combined_df.head())

print("\n--- Value counts for Primary_Condition in Augmented Data (Top 20) ---")
print(combined_df["Primary_Condition"].value_counts().nlargest(20))

print("\n--- Value counts for Primary_Condition in Original Data (Top 20) ---")
print(original_df["Primary_Condition"].value_counts().nlargest(20))


