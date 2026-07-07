import sqlite3
import json
import datetime
from pathlib import Path
import traceback
from src.utils.text_utils import clean_text

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        # Ensure database parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self):
        """Returns a sqlite3 connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes schema tables and default indexes."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS Users (
                    email TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('Student', 'Educator', 'Admin')),
                    login_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """)
                
                # Emotion_Records table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS Emotion_Records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    field TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    predicted_emotion TEXT NOT NULL,
                    secondary_emotion TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    emotion_scores TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    csv_logged INTEGER DEFAULT 0,
                    FOREIGN KEY (email) REFERENCES Users(email) ON DELETE CASCADE
                )
                """)
                
                # Indices for search performance and foreign key joints
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_email ON Emotion_Records(email)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_timestamp ON Emotion_Records(timestamp)")
                conn.commit()
        except Exception as e:
            print("Error initializing database tables:", e)
            traceback.print_exc()
        finally:
            conn.close()

    # User CRUD operations
    def create_user(self, email, name, password_hash, role="Student"):
        """Creates a new user in the database."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                created_at = datetime.datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO Users (email, name, password, role, login_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (email.lower().strip(), name.strip(), password_hash, role, 0, created_at)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # User already exists
            return False
        finally:
            conn.close()

    def get_user(self, email):
        """Fetches a user record by email."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users WHERE email = ?", (email.lower().strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def increment_login_count(self, email):
        """Increments the login counter for authentication tracking."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE Users SET login_count = login_count + 1 WHERE email = ?", 
                    (email.lower().strip(),)
                )
                conn.commit()
        finally:
            conn.close()

    def get_all_users(self):
        """Fetches all users registered in the system (for admin dashboard)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT email, name, role, login_count, created_at FROM Users")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # Emotion Record CRUD operations
    def save_emotion_record(self, email, field, input_text, predicted_emotion, 
                            secondary_emotion, confidence_score, model_used, 
                            ai_response, response_type, emotion_scores, csv_logged=0):
        """Saves a new analysis query log to the database."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                timestamp = datetime.datetime.now().isoformat()
                
                # Serialize emotion score probabilities to JSON string
                scores_json = json.dumps(emotion_scores)
                
                cursor.execute("""
                    INSERT INTO Emotion_Records (
                        email, field, input_text, predicted_emotion, secondary_emotion, 
                        confidence_score, model_used, ai_response, response_type, 
                        emotion_scores, timestamp, csv_logged
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    email.lower().strip(), field, input_text, predicted_emotion, 
                    secondary_emotion, confidence_score, model_used, ai_response, 
                    response_type, scores_json, timestamp, csv_logged
                ))
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_user_records(self, email):
        """Gets all records for a specific user, sorted by timestamp descending."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM Emotion_Records WHERE email = ? ORDER BY timestamp DESC", 
                (email.lower().strip(),)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_all_records(self):
        """Gets all records in the database (for admin / educator analytics dashboards)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Emotion_Records ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_ai_response(self, record_id, new_response, response_type="Gemini"):
        """Updates the AI advice response column for regeneration support."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE Emotion_Records SET ai_response = ?, response_type = ? WHERE record_id = ?",
                    (new_response, response_type, record_id)
                )
                conn.commit()
                return True
        finally:
            conn.close()

    def delete_record(self, record_id):
        """Deletes a record from history logs."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM Emotion_Records WHERE record_id = ?", (record_id,))
                conn.commit()
                return True
        finally:
            conn.close()

    def mark_records_as_csv_logged(self, record_ids):
        """Marks a set of records as successfully backed up to CSV."""
        if not record_ids:
            return
        placeholders = ",".join("?" for _ in record_ids)
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE Emotion_Records SET csv_logged = 1 WHERE record_id IN ({placeholders})",
                    tuple(record_ids)
                )
                conn.commit()
        finally:
            conn.close()
