import unittest
from src.utils.text_utils import clean_text, remove_stopwords, apply_keyword_boosting
from src.services.gemini_service import GeminiSupportService

class TestPipelineUtilities(unittest.TestCase):
    def test_text_cleaning(self):
        raw_text = "Hello!! I am studying Python-3 on http://google.com."
        cleaned = clean_text(raw_text)
        # Should be lowercased, URLs removed, special characters stripped, and spacing normalized
        self.assertEqual(cleaned, "hello i am studying python on")

    def test_stopword_removal(self):
        text = "this is a very simple check code"
        filtered = remove_stopwords(text)
        # 'this', 'is', 'a', 'very' are common stopwords
        self.assertNotIn("this", filtered)
        self.assertNotIn("is", filtered)
        self.assertIn("simple", filtered)
        self.assertIn("check", filtered)

    def test_rule_based_keyword_boosting(self):
        # Initial equal probabilities (sum = 1.0)
        initial_scores = {
            "Bored": 0.20,
            "Confident": 0.20,
            "Confused": 0.20,
            "Curious": 0.20,
            "Frustrated": 0.20
        }
        
        # Text containing strong frustration keywords
        text = "nothing is working compiler error syntax error debugging is a nightmare"
        boosted = apply_keyword_boosting(initial_scores, text)
        
        # Frustrated score should be boosted and be the highest probability class
        self.assertGreater(boosted["Frustrated"], boosted["Bored"])
        self.assertGreater(boosted["Frustrated"], 0.20)
        
        # Ensure re-normalized scores sum close to 1.0
        self.assertAlmostEqual(sum(boosted.values()), 1.0, places=5)

    def test_gemini_fallback_tutor_responses(self):
        service = GeminiSupportService("") # Pass empty key to trigger fallback
        
        fallback_text, resp_type = service.generate_guidance(
            field="Mathematics",
            problem_context="I don't understand eigenvalues and eigenvectors",
            primary_emotion="Confused",
            secondary_emotion="None",
            confidence_score=0.85
        )
        
        self.assertEqual(resp_type, "Predefined Template (Offline Fallback)")
        self.assertIn("### 1. Empathy Acknowledgment", fallback_text)
        self.assertIn("### 2. Learning Guidance", fallback_text)
        self.assertIn("### 3. Motivational Encouragement", fallback_text)
        self.assertIn("### 4. Next Steps", fallback_text)
        self.assertIn("eigenvalues and eigenvectors", fallback_text)

if __name__ == "__main__":
    unittest.main()
