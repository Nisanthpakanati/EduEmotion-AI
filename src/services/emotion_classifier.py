import traceback
import os
import json
import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.text_utils import clean_text, apply_keyword_boosting

# ---------------------------------------------------------------------------
# Emotion label mapping (must match training label order)
# ---------------------------------------------------------------------------
EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
EMOTION_MAP = {name: idx for idx, name in enumerate(EMOTION_CLASSES)}


# ---------------------------------------------------------------------------
# BiLSTM Architecture (must match train_bilstm.py exactly)
# ---------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_classes=5, num_layers=2, dropout=0.3):
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
        lstm_out, _ = self.lstm(embedded)
        out = torch.mean(lstm_out, dim=1)
        out = self.fc(self.dropout(out))
        return out


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class EmotionClassifierPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve base dir: this file is at src/services/emotion_classifier.py
        # so .parent.parent.parent == project root
        self.base_dir = Path(__file__).resolve().parent.parent.parent

        # Model paths
        self.bilstm_dir = self.base_dir / "models" / "bilstm"
        self.bert_dir   = self.base_dir / "models" / "bert"

        # Model objects
        self.bilstm_model    = None
        self.vocab           = None
        self.bert_model      = None
        self.bert_tokenizer  = None

        # Print startup diagnostics
        self._print_diagnostics()

        # Load models
        self.load_bilstm()
        self.load_bert()

        # Run a test prediction to confirm everything is wired up
        self._run_startup_test()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _print_diagnostics(self):
        print("\n" + "=" * 60)
        print("  EmotionClassifierPipeline — Startup Diagnostics")
        print("=" * 60)
        print(f"  Python          : {sys.version.split()[0]}")
        print(f"  PyTorch         : {torch.__version__}")
        print(f"  Device          : {self.device}")
        print(f"  CWD             : {os.getcwd()}")
        print(f"  Base dir        : {self.base_dir}")
        print(f"  BERT dir        : {self.bert_dir}")
        print(f"  BiLSTM dir      : {self.bilstm_dir}")
        print()

        # BERT files
        print("  BERT model files:")
        bert_files = [
            "config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"
        ]
        for fname in bert_files:
            exists = (self.bert_dir / fname).exists()
            mark = "OK" if exists else "MISSING"
            print(f"    {fname:35s} {mark}")

        print()
        print("  BiLSTM model files:")
        for fname in ["vocab.json", "model.pt"]:
            exists = (self.bilstm_dir / fname).exists()
            mark = "OK" if exists else "MISSING"
            print(f"    {fname:35s} {mark}")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Load BiLSTM
    # ------------------------------------------------------------------
    def load_bilstm(self):
        """Loads vocabulary and state dict for BiLSTM model."""
        vocab_path = self.bilstm_dir / "vocab.json"
        model_path = self.bilstm_dir / "model.pt"

        if not vocab_path.exists() or not model_path.exists():
            print("[BiLSTM] Model files not found — BiLSTM will be unavailable.")
            print(f"         vocab.json : {'OK' if vocab_path.exists() else 'MISSING'}")
            print(f"         model.pt   : {'OK' if model_path.exists() else 'MISSING'}")
            print("         Run training/train_bilstm.py to generate these files.\n")
            return

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.vocab = vocab_data["vocab"]

            self.bilstm_model = BiLSTMClassifier(vocab_size=len(self.vocab))
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.bilstm_model.load_state_dict(state_dict)
            self.bilstm_model.to(self.device)
            self.bilstm_model.eval()
            print(f"[BiLSTM] Loaded successfully. Vocab size: {len(self.vocab)}\n")

        except Exception:
            print("[BiLSTM] ERROR during loading:")
            traceback.print_exc()
            self.bilstm_model = None
            self.vocab = None

    # ------------------------------------------------------------------
    # Load BERT
    # ------------------------------------------------------------------
    def load_bert(self):
        """Loads fine-tuned DistilBERT tokenizer and model from local files only."""
        model_path = self.bert_dir

        if not (model_path / "config.json").exists():
            print("[BERT] config.json not found — BERT will be unavailable.")
            print(f"       Expected location: {model_path / 'config.json'}\n")
            return

        try:
            print(f"[BERT] Loading tokenizer from: {model_path}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            print("[BERT] Tokenizer loaded OK.")

            print(f"[BERT] Loading model weights from: {model_path}")
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("[BERT] Model loaded successfully.\n")

        except Exception:
            print("\n[BERT] ========== LOAD ERROR ==========")
            traceback.print_exc()
            print("[BERT] ===================================\n")
            self.bert_model     = None
            self.bert_tokenizer = None

    # ------------------------------------------------------------------
    # Startup test prediction
    # ------------------------------------------------------------------
    def _run_startup_test(self):
        """Runs a smoke-test prediction to confirm the pipeline is functional."""
        TEST_TEXT = "I am stressed about my exams and nothing makes sense."
        print("=" * 60)
        print("  Startup Test Prediction")
        print(f"  Input: \"{TEST_TEXT}\"")
        print("=" * 60)

        if self.bert_model is not None:
            try:
                result = self.predict_bert(TEST_TEXT)
                if result:
                    print(f"  [BERT]   Emotion={result['primary_emotion']:12s}  "
                          f"Confidence={result['confidence_score']:.1%}  "
                          f"Model={result['model_used']}")
            except Exception:
                print("  [BERT] Test prediction FAILED:")
                traceback.print_exc()
        else:
            print("  [BERT]   Not available (model not loaded).")

        if self.bilstm_model is not None:
            try:
                result = self.predict_bilstm(TEST_TEXT)
                if result:
                    print(f"  [BiLSTM] Emotion={result['primary_emotion']:12s}  "
                          f"Confidence={result['confidence_score']:.1%}  "
                          f"Model={result['model_used']}")
            except Exception:
                print("  [BiLSTM] Test prediction FAILED:")
                traceback.print_exc()
        else:
            print("  [BiLSTM] Not available (model files missing — train first).")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Preprocessing: BiLSTM
    # ------------------------------------------------------------------
    def _preprocess_bilstm_input(self, text, max_len=80):
        """Tokenizes and pads text into a tensor for BiLSTM inference."""
        cleaned = clean_text(text)
        words = cleaned.split()
        unk_idx = self.vocab.get("<UNK>", 1)
        indices = [self.vocab.get(w, unk_idx) for w in words]

        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))

        return torch.tensor([indices], dtype=torch.long).to(self.device)

    # ------------------------------------------------------------------
    # Predict: BiLSTM
    # ------------------------------------------------------------------
    def predict_bilstm(self, text):
        """Runs inference using the custom PyTorch BiLSTM model.
        Returns None if the model is not available."""
        if self.bilstm_model is None or self.vocab is None:
            return None

        try:
            cleaned = clean_text(text)
            tensor_input = self._preprocess_bilstm_input(text)

            with torch.no_grad():
                outputs = self.bilstm_model(tensor_input)
                probs = torch.softmax(outputs, dim=1).flatten().cpu().numpy()

            raw_scores = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}
            boosted_scores = apply_keyword_boosting(raw_scores, cleaned)
            return self._format_prediction(boosted_scores, "BiLSTM", text)

        except Exception:
            print("[BiLSTM] predict_bilstm() error:")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Predict: BERT
    # ------------------------------------------------------------------
    def predict_bert(self, text):
        """Runs inference using the fine-tuned DistilBERT model.
        Returns None if the model is not available."""
        if self.bert_model is None or self.bert_tokenizer is None:
            return None

        try:
            cleaned = clean_text(text)

            encoding = self.bert_tokenizer(
                cleaned,
                add_special_tokens=True,
                max_length=80,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )
                probs = torch.softmax(outputs.logits, dim=1).flatten().cpu().numpy()

            raw_scores = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}
            boosted_scores = apply_keyword_boosting(raw_scores, cleaned)
            return self._format_prediction(boosted_scores, "BERT", text)

        except Exception:
            print("[BERT] predict_bert() error:")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Format output
    # ------------------------------------------------------------------
    def _format_prediction(self, scores, model_name, original_text):
        """Formats raw score dict into the unified output schema."""
        sorted_emotions = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        primary_emotion    = sorted_emotions[0][0]
        confidence_score   = sorted_emotions[0][1]

        # Secondary emotion if it has >= 15% probability
        secondary_emotion = "None"
        if len(sorted_emotions) > 1 and sorted_emotions[1][1] >= 0.15:
            secondary_emotion = sorted_emotions[1][0]

        return {
            "primary_emotion":   primary_emotion,
            "secondary_emotion": secondary_emotion,
            "confidence_score":  float(confidence_score),
            "model_used":        model_name,
            "scores":            scores,
            "input_text":        original_text
        }
