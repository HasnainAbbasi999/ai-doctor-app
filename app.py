import torch
from transformers import pipeline
import streamlit as st

# Load the pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to diagnose symptoms using the model
def diagnose_symptoms(symptoms):
    medical_text = """
    Common causes of fever include viral infections, bacterial infections, and other inflammatory conditions.
    Headaches can be caused by stress, migraines, or dehydration.
    A cough can result from respiratory infections, allergies, or asthma. 
    """  # You can expand the context with more medical data

    query = f"What could be the possible cause of the symptoms: {symptoms}?"
    result = qa_pipeline(question=query, context=medical_text)
    answer = result['answer']
    
    return answer

# Streamlit App Structure
def main():
    st.title("AI Doctor Symptom Checker")

    # Ask the user to input symptoms
    symptoms = st.text_input("Enter your symptoms (e.g., fever, headache, cough):", "")
    
    if st.button('Diagnose'):
        if symptoms:
            # Run diagnosis
            diagnosis = diagnose_symptoms(symptoms)
            st.subheader("AI Doctor Diagnosis:")
            st.write(f"Based on your symptoms: {diagnosis}")
            
            # Provide a recommendation based on common symptoms
            if "fever" in symptoms or "cough" in symptoms:
                st.subheader("Recommendation:")
                st.write("You might need to consult a doctor for a thorough check-up. If symptoms persist or worsen, seek immediate medical attention.")
            else:
                st.subheader("Recommendation:")
                st.write("You can monitor your symptoms for now. If they persist or get worse, consult a healthcare provider.")
        else:
            st.write("Please enter your symptoms.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
