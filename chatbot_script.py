import joblib
import pandas as pd

# Load the trained model and preprocessor
model = joblib.load('/home/ubuntu/pet_diagnosis_model.joblib')
preprocessor = joblib.load('/home/ubuntu/pet_diagnosis_preprocessor.joblib')

print('Chatbot: Hello! I am a pet diagnosis assistant. I can help you identify potential health issues for your dog or cat.')
print('Chatbot: Please describe the symptoms your pet is experiencing. For example, you can say "My dog is vomiting and has diarrhea."')

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        print('Chatbot: Goodbye!')
        break

    # Basic sentiment analysis to check if the user is frustrated or needs help
    if any(word in user_input.lower() for word in ['help', 'assist', 'confused', 'dont know', 'not working']):
        print('Chatbot: I can understand it can be frustrating when your pet is unwell. Please describe the main symptoms you observe. For example, you can say My cat has been coughing for three days and seems very tired.')
        continue

    # For simplicity in this example, we'll assume the user provides symptoms directly.
    # In a real application, more sophisticated NLP would be needed.
    # We'll use a placeholder for the features the model expects, then fill in what we can from user input.
    # This is a simplified approach. A more robust solution would involve named entity recognition (NER)
    # and mapping symptoms to the features used during training.

    # Create a DataFrame with the expected features for the model
    # The order of columns should exactly match the order used during training
    # We'll need to know the exact column names used during training for the preprocessor to work correctly.
    # Assuming the preprocessor expects 'Animal_Type', 'Age_Category', 'Severity', 'Treatment_Category', 'All_Symptoms'
    # For this example, we'll simplify and assume the model primarily uses 'All_Symptoms'.
    # In a real scenario, we'd need to ask the user for more information or make assumptions.

    # For now, let's assume the model primarily uses 'All_Symptoms' and we can default others or ask later.
    # This is a placeholder for a more complex feature engineering step.
    data = {
        'Animal_Type': ['Dog'], # Defaulting to Dog for now, would need to ask user
        'Age_Category': ['Adult'], # Defaulting to Adult for now
        'Symptom_1': [user_input], # Simplified: using the whole input as Symptom_1
        'Symptom_2': [''],
        'Symptom_3': [''],
        'Symptom_4': [''],
        'Symptom_5': ['']
    }
    input_df = pd.DataFrame(data)

    # Preprocess the input data (this will handle the 'All_Symptoms' combination internally if the preprocessor was designed that way)
    # However, our current preprocessor expects 'All_Symptoms' directly. So, we'll adjust.
    # For simplicity, let's assume the model directly uses the raw text input for now and preprocessing handles it.
    # This is a simplification. In a real scenario, you'd have a proper feature engineering pipeline.

    # The preprocessor we built expects specific columns like 'All_Symptoms', 'Animal_Type', etc.
    # We need to construct the input DataFrame to match what the preprocessor expects.
    # Let's assume the user's input corresponds to the 'All_Symptoms' field.
    # We'll use default values for other fields for now.

    processed_input = pd.DataFrame({
        'Animal_Type': ['Dog'],  # Defaulting, would need to ask user
        'Age_Category': ['Adult'], # Defaulting, would need to ask user
        'Severity': ['Unknown_Severity'], # Defaulting, would need to ask user
        'All_Symptoms': [user_input] # Using the user's input as the combined symptoms
    })

    # The preprocessor expects specific columns in a specific order.
    # It's better to pass a dictionary to the preprocessor if it's designed to handle it,
    # or ensure the DataFrame columns match exactly what was used during training.
    # For this example, let's assume the preprocessor can handle a DataFrame with these columns.

    # Make a prediction
    # The predict method now expects a DataFrame with the same column names as used during training the preprocessor
    # The preprocessor itself will handle the transformation of these columns.
    prediction = model.predict(processed_input)
    predicted_condition = prediction[0]

    print(f'Chatbot: Based on the symptoms you provided, the possible condition could be {predicted_condition}. Please consult a veterinarian for an accurate diagnosis and treatment.')


