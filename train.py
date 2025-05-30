import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Load and prepare dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1', on_bad_lines='skip')
df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
df = df[['label', 'text']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenization (convert to list!)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Dataset class
class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = SMSDataset(train_encodings, list(train_labels))
test_dataset = SMSDataset(test_encodings, list(test_labels))

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./logs',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs'
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("model/distilbert_model")
tokenizer.save_pretrained("model/distilbert_model")

print("✅ Training complete. Model saved to model/distilbert_model")
