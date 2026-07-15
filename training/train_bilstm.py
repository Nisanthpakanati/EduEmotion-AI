import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
import json
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_emotions_dataset.csv"
BILSTM_OUT_DIR = BASE_DIR / "models" / "bilstm"
BILSTM_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion mapping
EMOTION_MAP = {"Bored": 0, "Confident": 1, "Confused": 2, "Curious": 3, "Frustrated": 4}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# Custom Dataset
class StudentEmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=80):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and numericalize
        words = clean_text(text).split()
        indices = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        
        # Truncate / Pad
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [0] * (self.max_len - len(indices)) # 0 is padding idx
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# BiLSTM Model Definition
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=5, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(embedded)
        # Take the mean of LSTM outputs across time steps
        out = torch.mean(lstm_out, dim=1)
        out = self.fc(self.dropout(out))
        return out

def train_model():
    # Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Train-test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])
    
    # Build vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in train_df["cleaned_text"]:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocab mapping
    vocab_path = BILSTM_OUT_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "classes": EMOTION_MAP}, f, indent=4)
    print(f"Saved vocabulary to: {vocab_path}")
    
    # Prepare datasets and loaders
    train_dataset = StudentEmotionDataset(
        train_df["text"].values, 
        train_df["emotion"].map(EMOTION_MAP).values, 
        vocab
    )
    val_dataset = StudentEmotionDataset(
        val_df["text"].values, 
        val_df["emotion"].map(EMOTION_MAP).values, 
        vocab
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = BiLSTMClassifier(vocab_size=len(vocab), num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training Loop
    epochs = 10
    best_acc = 0.0
    print("Starting BiLSTM training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = BILSTM_OUT_DIR / "model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved new best model checkpoint to {model_path}")
            
    # Final evaluation
    model.load_state_dict(torch.load(BILSTM_OUT_DIR / "model.pt"))
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(y.cpu().numpy())
            
    print("\nBiLSTM Final Classification Report:")
    target_names = [INV_EMOTION_MAP[i] for i in range(5)]
    print(classification_report(val_labels, val_preds, target_names=target_names))

if __name__ == "__main__":
    train_model()
