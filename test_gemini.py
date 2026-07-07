import sys, pathlib, os
sys.stdout.reconfigure(encoding='utf-8')
base = pathlib.Path(r'c:\Users\nisha\OneDrive\Desktop\emotion detection (internship)')
os.chdir(str(base))
sys.path.insert(0, str(base))

from src.services.emotion_classifier import EmotionClassifierPipeline
from src.services.gemini_service import GeminiSupportService
import config

print('=== END TO END TEST ===')
classifier = EmotionClassifierPipeline()

# Try with whatever is in config, but if it's empty, use a dummy key to force API attempt
api_key_to_test = config.GEMINI_API_KEY if config.GEMINI_API_KEY else 'dummy_invalid_api_key_to_force_attempt'
print('Initializing Gemini Support Service...')
gemini = GeminiSupportService(api_key_to_test)

student_query = "I have my Data Structures exam tomorrow. I am anxious and don't know where to begin."
academic_field = "Computer Science"

print('\n[1] BERT Emotion Prediction:')
bert_res = classifier.predict_bert(student_query)
if bert_res:
    print(f"Primary Emotion: {bert_res['primary_emotion']} (Confidence: {bert_res['confidence_score']:.1%})")
else:
    print('Failed to get BERT prediction.')

print('\n[2] Gemini Generation:')
if bert_res:
    ai_text, response_type = gemini.generate_guidance(
        academic_field, 
        student_query, 
        bert_res['primary_emotion'], 
        bert_res['secondary_emotion'], 
        bert_res['confidence_score']
    )
    print('\n[Result] Response Type:', response_type)
    print('[Result] AI Text preview:\n', ai_text[:300] + '...')
