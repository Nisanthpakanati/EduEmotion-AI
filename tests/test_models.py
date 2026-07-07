import unittest
import torch
from src.services.emotion_classifier import BiLSTMClassifier

class TestModelArchitectures(unittest.TestCase):
    def test_bilstm_dimensions(self):
        # Hyperparameters
        vocab_size = 1000
        embed_dim = 128
        hidden_dim = 128
        num_classes = 5
        batch_size = 4
        seq_length = 80
        
        # Instantiate model
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=2
        )
        
        # Create dummy batch input (integer token indices)
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Run forward pass
        outputs = model(dummy_input)
        
        # Check output tensor dimensions (should be batch_size x num_classes)
        self.assertEqual(outputs.shape, (batch_size, num_classes))
        
        # Ensure outputs are valid float numbers
        self.assertFalse(torch.isnan(outputs).any().item(), "Model output contains NaN values.")

if __name__ == "__main__":
    unittest.main()
