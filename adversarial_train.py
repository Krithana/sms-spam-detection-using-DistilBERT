import pandas as pd
import random
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer – assumes you've already fine-tuned and saved it
MODEL_PATH = "model/distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load SMS data
# NOTE: file might have some weird columns depending on source, hence the cleanup
try:
    raw_df = pd.read_csv("data/spam.csv", encoding='latin-1', on_bad_lines='skip')
except Exception as e:
    print("Failed to load CSV:", e)
    raise

# Fix up column names – some datasets have v1, v2 instead of label/text
if 'v1' in raw_df.columns and 'v2' in raw_df.columns:
    df = raw_df[['v1', 'v2']]
    df.columns = ['label', 'text']
elif raw_df.shape[1] >= 2:
    df = raw_df.iloc[:, :2]
    df.columns = ['label', 'text']
else:
    raise ValueError("❌ Couldn't detect valid columns in spam.csv")

# Clean the data
df = df.dropna()  # Remove missing entries just in case
df = df[df['text'].apply(lambda x: isinstance(x, str))]  # Filter out non-strings (edge cases)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Just re-map string labels to numeric

# Simulate an adversarial typo (not too fancy – just a basic character switch)
def add_typos(msg):
    words = msg.split()
    if len(words) < 3:
        return msg  # not worth corrupting super short texts
    index_to_mess = random.randint(0, len(words) - 1)
    word = words[index_to_mess]
    if len(word) > 3:
        # Random char inserted – could definitely be smarter, but keeping it quick
        messed_up = word[:2] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[3:]
        words[index_to_mess] = messed_up
    return " ".join(words)

# Make a prediction – returns both class and spam probability
def get_prediction(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
    return predicted_class, float(probs[1])

# Let's run a few tests on random samples
results = []
sample_set = df.sample(10, random_state=42)  # same sample for reproducibility

for idx, row in sample_set.iterrows():
    original_text = row['text']
    adversarial_text = add_typos(original_text)

    pred_orig, prob_orig = get_prediction(original_text)
    pred_adv, prob_adv = get_prediction(adversarial_text)

    results.append({
        "Original Text": original_text,
        "Original Label": "Spam" if row['label'] == 1 else "Ham",
        "Original Prediction": "Spam" if pred_orig == 1 else "Ham",
        "Original Prob": round(prob_orig, 3),
        "Adversarial Text": adversarial_text,
        "Adversarial Prediction": "Spam" if pred_adv == 1 else "Ham",
        "Adversarial Prob": round(prob_adv, 3),
        "Changed": "Yes" if pred_orig != pred_adv else "No"
    })

# Save everything to a CSV so it can be visualized later
result_df = pd.DataFrame(results)
result_df.to_csv("adversarial_results.csv", index=False)

print("✅ Adversarial testing complete! Results saved to 'adversarial_results.csv'")
