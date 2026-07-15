# pyrefly: ignore [missing-import]
import bcrypt
# pyrefly: ignore [missing-import]
import streamlit as st
import re

def hash_password(password):
    """Hashes a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password, hashed_password):
    """Verifies a plaintext password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def is_valid_email(email):
    """Validates email format using regex."""
    regex = r"^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return bool(re.match(regex, email))

def register_user(db_manager, email, name, password, role="Student"):
    """
    Registers a new user by validation, hashing the password, 
    and storing the details in the database.
    """
    email = email.strip().lower()
    name = name.strip()
    
    # Input validation checks
    if not email or not name or not password:
        return False, "All fields are required."
        
    if not is_valid_email(email):
        return False, "Invalid email address format."
        
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
        
    if role not in ["Student", "Educator", "Admin"]:
        return False, "Invalid user role specified."
        
    # Check if user already exists
    existing = db_manager.get_user(email)
    if existing:
        return False, "An account with this email already exists."
        
    # Hash password
    pw_hash = hash_password(password)
    
    # Create user
    success = db_manager.create_user(email, name, pw_hash, role)
    if success:
        return True, "Registration successful! You can now log in."
    else:
        return False, "Registration failed due to a database error."

def authenticate_user(db_manager, email, password):
    """
    Validates user credentials against the database.
    Increments the login counter if successful.
    """
    email = email.strip().lower()
    if not email or not password:
        return False, "Email and password are required."
        
    user = db_manager.get_user(email)
    if not user:
        return False, "Incorrect email or password."
        
    if verify_password(password, user["password"]):
        # Update login counter
        db_manager.increment_login_count(email)
        return True, user
    else:
        return False, "Incorrect email or password."

def initialize_session():
    """Initializes Streamlit session state properties on load."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "current_page" not in st.session_state:
        # Default starting page is landing
        st.session_state.current_page = "Home"

def login_user(user_data):
    """Stores user details into Streamlit session state."""
    st.session_state.logged_in = True
    st.session_state.user_email = user_data["email"]
    st.session_state.user_name = user_data["name"]
    st.session_state.user_role = user_data["role"]
    st.session_state.current_page = "Workspace"

def logout_user():
    """Clears all authentication properties from Streamlit session state."""
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_name = None
    st.session_state.user_role = None
    st.session_state.current_page = "Home"
