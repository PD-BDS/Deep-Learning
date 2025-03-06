import streamlit as st
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Mental Health Classifier", layout="centered")

# Load tokenizer and model
model_path = "best_model.pth"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=7)

try:
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only= True))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
st.write("Enter a statement, and the model will classify it into one of the mental health conditions.")

# User input
user_input = st.text_area("Enter your text here:")

if st.button("See Mental Status"):
    if user_input.strip():
        try:
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()

            # label mapping
            label_map = {
                0: "Normal",
                1: "Depression",
                2: "Suicidal",
                3: "Anxiety",
                4: "Bipolar",
                5: "Stress",
                6: "Personality Disorder"
            }

            st.write(f"**Predicted Mental Health Condition:** {label_map.get(prediction, 'Unknown')}")
        except Exception as e:
            st.error(f"Error during classification: {e}")
    else:
        st.warning("Please enter some text for classification.")
