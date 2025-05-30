from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load up the model – assumes it's fine-tuned and saved in the local directory
MODEL_PATH = "model/distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Predict if a given message is spam or not
def classify_sms(message):
    # Tokenize the message to the format the model expects
    encoded_input = tokenizer(message, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():  # inference only
        result = model(**encoded_input)

    logits = result.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Debug line if needed: print("Raw logits:", logits)
    
    return "Spam" if predicted_class == 1 else "Ham"

if __name__ == "__main__":
    print("📩 SMS Spam Classifier\n")
    user_input = input("Enter an SMS message: ")
    outcome = classify_sms(user_input)
    print("Prediction:", outcome)
