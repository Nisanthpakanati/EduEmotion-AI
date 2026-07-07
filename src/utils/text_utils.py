import re

# Standalone list of common English stopwords to ensure 100% offline robustness
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now"
}

# Emotion-specific keyword triggers for rule-based boosting
BOOST_KEYWORDS = {
    "Bored": [
        "bored", "boring", "sleepy", "dull", "dry", "tiresome", "monotonous", 
        "tedious", "slow pace", "snail's pace", "pointless", "rote", 
        "uninteresting", "disengaged", "falling asleep", "checked out"
    ],
    "Confident": [
        "confident", "mastered", "mastery", "easy", "breeze", "simple", 
        "finished", "perfect", "understand", "clear", "straightforward", 
        "got this", "got it down", "productive", "clicking", "solved"
    ],
    "Confused": [
        "confused", "lost", "stuck", "struggling", "don't understand", 
        "don't get", "makes no sense", "unsure", "baffled", "unclear", 
        "how does", "what is the difference", "explain", "baffling", 
        "get mixed up", "what does it mean", "not clear"
    ],
    "Curious": [
        "curious", "fascinating", "want to know", "want to explore", 
        "read more", "interested", "how does it work", "what happens if", 
        "deep dive", "under the hood", "real-world application", "eager", 
        "wondering", "implications", "spark my interest", "fascinated"
    ],
    "Frustrated": [
        "frustrated", "annoyed", "mad", "hate", "bug", "compiler error", 
        "syntax error", "crashing", "crashes", "hanging", "hangs", 
        "useless error", "throwing my laptop", "spent hours", "spent the whole weekend", 
        "nothing works", "give up", "driving me crazy", "annoying", "nightmare"
    ]
}

def clean_text(text):
    """Normalize, lowercase, and remove special characters from text."""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Normalize spacing
    return " ".join(text.split())

def remove_stopwords(text):
    """Remove stopwords using our offline vocabulary set."""
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered)

def apply_keyword_boosting(scores, cleaned_text):
    """
    Scans the text for emotion keywords. For each match, we add a weight 
    to the prediction probability, and then re-normalize all scores to sum to 1.0.
    """
    boosted_scores = scores.copy()
    has_boost = False
    
    # Calculate boost additions
    for emotion, keywords in BOOST_KEYWORDS.items():
        count = 0
        for kw in keywords:
            # Match whole phrases/words using simple substring or word boundary checks
            if kw in cleaned_text:
                count += 1
        
        if count > 0:
            # Add 0.10 for each matched keyword (cap boost impact to maintain model integrity)
            boost_amt = min(0.30, count * 0.10)
            boosted_scores[emotion] += boost_amt
            has_boost = True
            
    if has_boost:
        # Re-normalize to sum to 1.0
        total_score = sum(boosted_scores.values())
        if total_score > 0:
            boosted_scores = {k: v / total_score for k, v in boosted_scores.items()}
            
    return boosted_scores
