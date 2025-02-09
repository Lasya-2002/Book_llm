import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate

# Load dataset
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/google_books_api_dataset/Google_books_api_dataset.csv")

# Combine all columns into a single text input
def merge_features(row):
    return f"Title: {row['Title']} Description: {row['Description']} Authors: {row['Authors']} " \
           f"Publisher: {row['Publisher']} Categories: {row['Categories']} Ratings: {row['Average Rating']} " \
           f"PageCount: {row['Page Count']} Language : {row['Language']} ({row['Ratings Count']} ratings)"

df['text'] = df.apply(merge_features, axis=1)

# Convert ratings to classification labels
def categorize_rating(rating):
    if rating >= 4.5:
        return 2  # High rating
    elif rating >= 3.0:
        return 1  # Medium rating
    else:
        return 0  # Low rating

df['label'] = df['Average Rating'].fillna(0).apply(categorize_rating)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_function, batched=True)
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_function, batched=True)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_finetuned_books",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10000,
    save_total_limit=2,
)

# Load accuracy metric
accuracy_metric = evaluate.load("accuracy")

# Define compute_metrics function
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get predicted class
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Track accuracy
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./bert_finetuned_books")
tokenizer.save_pretrained("./bert_finetuned_books")
