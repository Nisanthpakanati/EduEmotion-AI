import unittest
import os
import tempfile
from pathlib import Path
from src.database.db_manager import DatabaseManager
from src.auth.session_manager import register_user, authenticate_user, hash_password, verify_password

class TestDatabaseAndAuth(unittest.TestCase):
    def setUp(self):
        # Create a temporary test database file within the workspace
        self.db_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.test_db_path = str(self.db_dir / "test_emotion_platform.db")
        self.db = DatabaseManager(self.test_db_path)

    def tearDown(self):
        # Clean up temporary database files
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass

    def test_user_creation_and_auth(self):
        # Test registration
        success, msg = register_user(
            self.db, 
            "test_student@edu.com", 
            "Test Student", 
            "secure_pass123", 
            "Student"
        )
        self.assertTrue(success, f"Failed registration: {msg}")
        
        # Test duplicate registration block
        success2, msg2 = register_user(
            self.db, 
            "test_student@edu.com", 
            "Test Student Duplicate", 
            "another_pass", 
            "Student"
        )
        self.assertFalse(success2)
        self.assertIn("already exists", msg2)
        
        # Test login verification
        auth_success, user_data = authenticate_user(self.db, "test_student@edu.com", "secure_pass123")
        self.assertTrue(auth_success)
        self.assertEqual(user_data["name"], "Test Student")
        self.assertEqual(user_data["role"], "Student")
        
        # Test incorrect password login block
        auth_fail, fail_msg = authenticate_user(self.db, "test_student@edu.com", "wrong_pass")
        self.assertFalse(auth_fail)

    def test_save_and_update_records(self):
        # Create user
        register_user(self.db, "test_user@edu.com", "Tester", "pass123", "Student")
        
        # Save record
        scores = {"Bored": 0.05, "Confident": 0.05, "Confused": 0.80, "Curious": 0.05, "Frustrated": 0.05}
        record_id = self.db.save_emotion_record(
            email="test_user@edu.com",
            field="Computer Science",
            input_text="I am lost on pointer arithmetic",
            predicted_emotion="Confused",
            secondary_emotion="None",
            confidence_score=0.80,
            model_used="BERT",
            ai_response="### Empathy\nI see you are confused...",
            response_type="Template",
            emotion_scores=scores
        )
        
        self.assertIsNotNone(record_id)
        self.assertGreater(record_id, 0)
        
        # Retrieve user records
        user_recs = self.db.get_user_records("test_user@edu.com")
        self.assertEqual(len(user_recs), 1)
        self.assertEqual(user_recs[0]["predicted_emotion"], "Confused")
        self.assertEqual(user_recs[0]["model_used"], "BERT")
        
        # Update response (regeneration step)
        success = self.db.update_ai_response(record_id, "### Empathy\nNew response text", "Gemini")
        self.assertTrue(success)
        
        updated_recs = self.db.get_user_records("test_user@edu.com")
        self.assertEqual(updated_recs[0]["ai_response"], "### Empathy\nNew response text")
        self.assertEqual(updated_recs[0]["response_type"], "Gemini")

if __name__ == "__main__":
    unittest.main()
