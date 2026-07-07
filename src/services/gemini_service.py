import os
import time
import traceback
import google.generativeai as genai
from src.utils.text_utils import clean_text

# Predefined high-quality offline templates for each emotion
OFFLINE_TEMPLATES = {
    "Bored": {
        "acknowledgment": "It is completely normal to feel disengaged or bored when studying concepts that seem abstract or repetitive.",
        "guidance": "To make learning {field} more active, try translating theoretical concepts into immediate practical tasks. For instance, if you are learning a dry algorithm, write a small script to visualize its step-by-step process, or research a real-world case study where this exact topic saved a company millions.",
        "motivation": "Remember that dry foundations are the building blocks for creating incredibly exciting and complex systems later. Keep going!",
        "next_steps": "1. Set a 20-minute focus timer (Pomodoro technique).\n2. Write down one practical application of this concept in your field.\n3. Try to explain this concept to a friend using an analogy."
    },
    "Confident": {
        "acknowledgment": "Fantastic! Having a warm, confident grasp of your study topic is an excellent milestone.",
        "guidance": "Since you have mastered the basics, challenge yourself by investigating advanced optimization techniques, refactoring your solutions for maximum efficiency, or writing comprehensive unit tests to ensure edge cases are handled.",
        "motivation": "Maintain this excellent momentum and curiosity. Use this energy to tackle the next harder topic in {field}!",
        "next_steps": "1. Find a challenge problem or advanced repository related to this topic.\n2. Help a classmate or explain the concept on a forum.\n3. Move on to the next module in your syllabus."
    },
    "Confused": {
        "acknowledgment": "Confusion is not a sign of failure—it is the active process of your brain building new neural connections. Acknowledge the block and take it step by step.",
        "guidance": "Let's break down the problem. Write out the input and output variables. Trace the flow of logic step-by-step on a piece of paper. Try to isolate the exact line, symbol, or equation where the behavior departs from what you expect.",
        "motivation": "Every senior professional in {field} has been confused by this exact concept. You are on the right path!",
        "next_steps": "1. Write down what you *do* understand about the problem first.\n2. Look up a simple visual tutorial or documentation page for this specific function/concept.\n3. Ask a peer or mentor with a clear description of the block."
    },
    "Curious": {
        "acknowledgment": "It is wonderful to see your curiosity sparked! Asking deeper questions is how you move from a student to an expert.",
        "guidance": "To feed your curiosity in {field}, investigate how this concept handles extreme scale, how it is implemented under the hood in compiler/hardware logic, or read recent academic publications describing enhancements to this theory.",
        "motivation": "Never lose this drive to ask 'why' and 'how'. It is your greatest asset as a learner and creator.",
        "next_steps": "1. Read the official source code or advanced standard documentation for this topic.\n2. Design a small experiment to test the limits of this concept.\n3. Bookmark 2 research papers or articles related to this topic."
    },
    "Frustrated": {
        "acknowledgment": "We hear you. Spending hours debugging a single error or hitting a roadblock is incredibly draining and frustrating.",
        "guidance": "Let's do a hard reset. Step away from your desk for five minutes. When you return, read the compiler/error logs line-by-line starting from the top. Implement print statements or debug logs directly before the crash point to inspect the exact runtime state.",
        "motivation": "The feeling of relief and victory when you solve this frustration is what makes engineering and study so rewarding. You are close to the solution!",
        "next_steps": "1. Take a 5-minute physical break (walk around, hydrate).\n2. Explain the code or formula to a 'rubber duck' (explain it out loud word-by-word).\n3. Check for trivial typos, syntax mismatch, or environment file setup issues."
    }
}

class GeminiSupportService:
    def __init__(self, api_key=""):
        self.api_key = api_key if api_key else os.getenv("GEMINI_API_KEY", "")
        self.client_initialized = False
        
        # 1 & 2. Verify key read and print masked
        if self.api_key:
            masked = self.api_key[:6] + "*" * (len(self.api_key) - 12) + self.api_key[-6:] if len(self.api_key) >= 12 else "***"
            print(f"Gemini API Key Loaded: {masked}")
            try:
                genai.configure(api_key=self.api_key)
                self.client_initialized = True
                # 3. Print initialization success
                print("Gemini Client Initialized successfully.")
            except Exception:
                print("[Gemini] Error during genai.configure():")
                traceback.print_exc()
        else:
            print("No Gemini API key found (key is empty). Using offline response templates as fallback.")

    def generate_guidance(self, field, problem_context, primary_emotion, secondary_emotion, confidence_score):
        """
        Generates empathetic learning support. Tries Gemini API with retry logic,
        and falls back to pre-defined offline templates if unavailable.
        """
        prompt = self._construct_prompt(field, problem_context, primary_emotion, secondary_emotion, confidence_score)
        
        if self.client_initialized:
            # 11. Model fallback logic
            models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
            
            for model_name in models_to_try:
                for attempt in range(2): # 2 attempts per model
                    # 4. Print exact model name
                    print(f"Attempting API call with model: {model_name} (Attempt {attempt + 1})")
                    # 5. Before every API call print:
                    print("Sending request to Gemini...")
                    
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        if response and response.text:
                            # 6. After the call print success
                            print("Response received successfully")
                            return response.text.strip(), "Gemini API"
                    except Exception as e:
                        # 6, 7, 8. Remove silent blocks, print error code and traceback
                        print(f"HTTP/API error occurred while calling {model_name}!")
                        
                        # Extract error code if available (google API core exceptions)
                        error_code = getattr(e, "code", None)
                        if error_code:
                            print(f"Error Code: {error_code}")
                        if hasattr(e, "message"):
                            print(f"Error Message: {e.message}")
                            

                        traceback.print_exc()
                        time.sleep(2 ** attempt)
                        
            print("All Gemini API calls and model fallbacks failed. Falling back to offline templates.")
            
        # Fallback template formatting
        return self._generate_fallback(field, primary_emotion), "Offline Fallback (Quota Exceeded)"

    def _construct_prompt(self, field, problem, primary, secondary, confidence):
        """Constructs a structured context-aware prompt for the LLM."""
        prompt = f"""
You are a senior, highly empathetic AI Tutor and Learning Coach. A student studying the field of {field} has run into a learning block and described it as:
"{problem}"

Our classifier has detected their emotional state:
- Primary Emotion: {primary} (Confidence: {confidence:.1%})
- Secondary Emotion: {secondary}

Please write a supportive response structured into four clearly labeled sections. Use markdown headings for each section:

### 1. Empathy Acknowledgment
Acknowledge their feeling of {primary} (and {secondary} if relevant) in a warm, non-judgmental, and validating tone. Make them feel heard.

### 2. Learning Guidance
Provide concrete, step-by-step guidance specific to the field of {field} to help them overcome this block. Provide a simple code snippet, example, or conceptual breakdown where appropriate.

### 3. Motivational Encouragement
Give them inspiring and motivational words to rebuild their focus and grit.

### 4. Next Steps
Suggest 2-3 specific, actionable next steps they should take *right now* to make progress.

Keep the advice clear, actionable, and formatted in clean markdown.
"""
        return prompt

    def _generate_fallback(self, field, primary_emotion):
        """Constructs a fallback response from local templates when Gemini is offline."""
        template = OFFLINE_TEMPLATES.get(primary_emotion, OFFLINE_TEMPLATES["Confused"])
        
        # Insert field and topic details dynamically
        ack = template["acknowledgment"]
        guidance = template["guidance"].format(field=field)
        motivation = template["motivation"].format(field=field)
        steps = template["next_steps"]
        
        fallback_text = f"""### 1. Empathy Acknowledgment
{ack}

### 2. Learning Guidance (Offline Template)
{guidance}

### 3. Motivational Encouragement
{motivation}

### 4. Next Steps
{steps}"""
        return fallback_text
