# Mental Health Status Classifier

## Model Description
This model is fine-tuned on a mental health dataset to classify text into 7 categories:  
- Normal  
- Depression  
- Suicidal  
- Anxiety  
- Bipolar  
- Stress  
- Personality Disorder  

## Training Details
- **Base Model:** BERT (bert-base-uncased)  
- **Dataset:** Custom mental health text dataset  
- **Fine-tuning Epochs:** 5  
- **Batch Size:** 16  
- **Optimizer:** AdamW  
- **Learning Rate:** 2e-5  

## Evaluation Metrics
- **Train Accuracy:** 96%
- **Validation Accuracy:** 77%
- **Test Accuracy:** 77%

## Intended Use
This model is designed for research and awareness purposes related to mental health classification. It **does not** replace professional diagnosis.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "piyaldey/Mental_Health_Status_Classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I feel very anxious and worried all the time."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
