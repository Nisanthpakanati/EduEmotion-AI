import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_emotions_dataset.csv"
BERT_OUT_DIR = BASE_DIR / "models" / "bert"
BERT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion mapping
EMOTION_MAP = {"Bored": 0, "Confident": 1, "Confused": 2, "Curious": 3, "Frustrated": 4}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# Custom Dataset for BERT
class StudentEmotionBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=80):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # Load model configuration and tokenizer
    # We use distilbert-base-uncased which is lightweight, robust, and trains quickly.
    model_name = "distilbert-base-uncased"
    print(f"Loading pre-trained tokenizer and model config: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )
    config = AutoConfig.from_pretrained(model_name, num_labels=5)
    
    # Map label names in config
    config.id2label = {str(i): name for i, name in INV_EMOTION_MAP.items()}
    config.label2id = {name: i for name, i in EMOTION_MAP.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    
    # Load and split dataset
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])
    
    train_dataset = StudentEmotionBERTDataset(
        train_df["text"].values,
        train_df["emotion"].map(EMOTION_MAP).values,
        tokenizer
    )
    val_dataset = StudentEmotionBERTDataset(
        val_df["text"].values,
        val_df["emotion"].map(EMOTION_MAP).values,
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    # Training
    epochs = 3 # Small number of epochs since student dataset patterns are highly structured
    print("Starting BERT fine-tuning...")
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item() * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save model, config and tokenizer files
            model.save_pretrained(BERT_OUT_DIR)
            tokenizer.save_pretrained(BERT_OUT_DIR)
            print(f"  --> Saved new best BERT model weights to {BERT_OUT_DIR}")
            
    # Final evaluation
    model = AutoModelForSequenceClassification.from_pretrained(BERT_OUT_DIR).to(device)
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
            
    print("\nBERT Final Classification Report:")
    target_names = [INV_EMOTION_MAP[i] for i in range(5)]
    print(classification_report(val_labels, val_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))

if __name__ == "__main__":
    train_model()
