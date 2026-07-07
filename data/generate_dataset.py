import csv
import random
import os
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Academic fields
FIELDS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry", 
    "Biology", "Engineering", "Business", "Literature", 
    "History", "Psychology"
]

# Base query templates for each emotion
TEMPLATES = {
    "Bored": [
        "This lecture on {topic} is so dry, I'm falling asleep.",
        "Why do we need to memorize these {topic} formulas? It feels so pointless.",
        "I've been listening to this talk on {topic} for an hour and nothing is happening.",
        "The class on {topic} is moving at a snail's pace. I'm so disengaged.",
        "I find {topic} incredibly tedious. It is just rote repetition.",
        "This {topic} homework is extremely monotonous. I can't stay focused.",
        "Another lecture on {topic}? I'd rather do anything else right now.",
        "Learning about {topic} feels like watching paint dry. Zero excitement.",
        "The professor's slides on {topic} are just walls of text. So uninteresting.",
        "I don't see the relevance of {topic} to real life. It is just empty theory.",
        "It's hard to pay attention to {topic} when it's taught this way.",
        "I keep checking the clock in my {topic} class. Time is standing still.",
        "This chapter on {topic} has no practical examples, just boring definitions.",
        "This topic is very dry and boring.",
        "I am losing interest in this class because {topic} is not engaging.",
        "This is just repetitive exercises in {topic}. I am totally checked out.",
        "I'm struggle to stay awake during {topic} discussions.",
        "This subject is too slow for me, especially the part about {topic}.",
        "We are just reading the textbook for {topic}. What a waste of time.",
        "I have no motivation to study {topic}. It is completely uninspiring."
    ],
    "Confident": [
        "I fully understand {topic} now and completed all the exercises easily!",
        "The test on {topic} was a breeze. I got a perfect score!",
        "I feel completely ready to explain {topic} to anyone who is struggling.",
        "This assignment on {topic} was simple. I finished it in ten minutes.",
        "I've mastered the core concepts of {topic}. Bring on the advanced stuff!",
        "My solution for the {topic} problem is clean, efficient, and working perfectly.",
        "I'm making great progress in {topic}. It's all clicking so fast.",
        "I solved the entire {topic} worksheet without looking at any hints.",
        "I feel confident about my grasp of {topic} ahead of the midterm.",
        "Implementing {topic} was straightforward. I knew exactly what to do.",
        "This module on {topic} was very easy to digest.",
        "I have a solid understanding of how {topic} works under the hood.",
        "My code for {topic} worked on the first run. I'm really getting good at this.",
        "I've got this {topic} topic completely under control.",
        "The explanation of {topic} was clear, and I implemented the model instantly.",
        "I don't have any doubts about {topic} anymore. I've got it down.",
        "I feel excellent about my performance in {topic} this week.",
        "I successfully solved the challenge problem on {topic} in record time.",
        "I am confident that my approach to {topic} is correct.",
        "This has been a very productive study session. I completely get {topic}."
    ],
    "Confused": [
        "I'm completely lost on {topic}. Can someone explain it in simple terms?",
        "I don't understand the difference between these two algorithms in {topic}.",
        "The lecture on {topic} went way too fast. I have no idea what's going on.",
        "I am stuck on this {topic} question. The instructions are very unclear.",
        "I keep getting confused when trying to apply the formula for {topic}.",
        "I'm really struggling to grasp the core concept of {topic}.",
        "Can someone explain how {topic} works? I am totally confused.",
        "I don't get the relationship between these variables in {topic}.",
        "Why does {topic} behave this way? It seems very counterintuitive.",
        "I have read this section on {topic} three times and still don't understand it.",
        "I'm unsure about the next steps in my {topic} analysis.",
        "I need help with {topic}. I'm getting mixed up with the definitions.",
        "The textbook explanation of {topic} is extremely confusing and dense.",
        "I am baffled by this error in {topic}. What does it mean?",
        "I don't understand the underlying theory behind {topic}.",
        "This concept in {topic} makes absolutely no sense to me.",
        "I'm having trouble understanding how to set up this {topic} model.",
        "Can someone break down {topic} step-by-step? I am lost.",
        "I'm confused about why we use this particular method for {topic}.",
        "I am struggling with the fundamentals of {topic}. It's not clear at all."
    ],
    "Curious": [
        "How does {topic} scale to real-world datasets? I'd love to know more.",
        "I'm curious to see what happens if we change the parameters in {topic}.",
        "Are there any advanced applications of {topic} in modern industry?",
        "This lecture on {topic} was fascinating! Where can I read more about it?",
        "I want to explore the history and development of {topic} theories.",
        "What is the mathematical connection between {topic} and other fields?",
        "I'm interested in working on a research project related to {topic}.",
        "How do researchers solve the open problems in {topic}?",
        "Can we apply {topic} concepts to solve this other problem?",
        "This is an interesting perspective on {topic}. How does it compare to others?",
        "I would love to learn more about the deep mechanisms of {topic}.",
        "What are the best resources to get a deeper understanding of {topic}?",
        "I'm wondering how {topic} will evolve with new technology.",
        "This concept in {topic} sparks my interest. What are the key papers on this?",
        "How does the model for {topic} work when inputs are noisy?",
        "I want to test if {topic} works under extreme constraints.",
        "Is it possible to combine {topic} with neural networks?",
        "I am eager to dive deeper into {topic} and find out how it handles edge cases.",
        "What is the intuition behind this specific proof in {topic}?",
        "I'd love to discuss the wider implications of {topic} on society."
    ],
    "Frustrated": [
        "I've been trying to solve this {topic} bug for hours and it's still failing!",
        "This compiler error in {topic} is driving me crazy! It makes no sense.",
        "I am so frustrated with {topic}. Nothing is working the way it should.",
        "I've spent the whole weekend on {topic} and made absolutely zero progress.",
        "Why is the setup for {topic} so incredibly difficult and poorly documented?",
        "This error message in {topic} is completely useless. I am stuck.",
        "I feel like throwing my laptop out the window. {topic} is so annoying.",
        "I have followed every step of the {topic} tutorial and it still crashes.",
        "This {topic} assignment is a nightmare. It is way too hard.",
        "I am tired of getting stuck on basic syntax errors in {topic}.",
        "I've tried everything to make this {topic} code run, but it keeps hanging.",
        "My {topic} results are completely wrong and I don't know how to fix them.",
        "This is so annoying. I keep losing my work because of this {topic} tool.",
        "I'm about to give up on this {topic} task. It is just too frustrating.",
        "Why does {topic} have to be so complicated? This is exhausting.",
        "I've asked for help twice and still can't solve this {topic} problem.",
        "This constant trial-and-error with {topic} is draining my energy.",
        "The server for {topic} keeps crashing and losing my connection. So mad.",
        "I hate dealing with these dependencies in {topic}. It's a massive waste of time.",
        "Nothing works! I am completely stuck on this {topic} integration."
    ]
}

# Field-specific topics for template insertion
TOPICS = {
    "Computer Science": [
        "recursion", "binary search trees", "gradient descent", "backpropagation",
        "pointer references", "memory allocation", "SQL joins", "multithreading",
        "API routing", "Docker containers", "merge sort", "regular expressions"
    ],
    "Mathematics": [
        "linear transformations", "partial derivatives", "matrix multiplication",
        "integrals", "eigenvalues and eigenvectors", "Bayes theorem",
        "differential equations", "probability distributions", "geometric proofs",
        "vector spaces", "Fourier series", "limits and continuity"
    ],
    "Physics": [
        "quantum entanglement", "thermodynamic laws", "special relativity",
        "electromagnetic induction", "projectile motion", "Schrodinger equation",
        "gravitational fields", "circuit diagrams", "angular momentum",
        "wave-particle duality", "nuclear fission", "fluid dynamics"
    ],
    "Chemistry": [
        "organic synthesis", "covalent bonding", "chemical equilibrium",
        "stoichiometric calculations", "acid-base titration", "periodic trends",
        "molecular orbitals", "thermochemistry", "enzyme kinetics",
        "redox reactions", "gas laws", "spectroscopy"
    ],
    "Biology": [
        "photosynthesis pathways", "mitosis and meiosis", "protein synthesis",
        "natural selection", "DNA replication", "action potentials",
        "homeostasis", "ecological niches", "cellular respiration",
        "gene splicing", "immune response", "phylogenetic trees"
    ],
    "Engineering": [
        "stress-strain analysis", "heat transfer equations", "feedback control loops",
        "fluid mechanics", "CAD modeling", "signal processing",
        "structural load calculations", "circuit simulation", "material strength",
        "aerodynamics", "thermodynamic cycles", "digital logic gates"
    ],
    "Business": [
        "financial accounting", "marketing strategies", "supply chain logistics",
        "microeconomic models", "balance sheets", "portfolio optimization",
        "market segmentation", "cost-benefit analysis", "macroeconomics",
        "organizational behavior", "corporate finance", "consumer psychology"
    ],
    "Literature": [
        "literary analysis", "metaphorical structures", "character development",
        "thematic analysis", "poetic meters", "narrative voices",
        "historical contexts", "structuralism", "rhetorical devices",
        "post-colonial literature", "genre conventions", "symbolism"
    ],
    "History": [
        "French Revolution", "Cold War diplomacy", "industrialization",
        "treaty negotiations", "ancient civilizations", "colonialism",
        "economic depressions", "social movements", "military strategies",
        "political revolutions", "renaissance art", "historical sources"
    ],
    "Psychology": [
        "cognitive dissonance", "classical conditioning", "neurotransmitters",
        "developmental stages", "defense mechanisms", "perception thresholds",
        "social influence", "personality traits", "memory retrieval",
        "behavior modification", "mental health disorders", "research ethics"
    ]
}

def generate_queries():
    dataset = []
    # Target size: 2500 entries (approx. 500 per emotion)
    target_count = 2500
    
    # We will loop and generate entries
    emotions = list(TEMPLATES.keys())
    
    # Track how many we've generated for each emotion
    counts = {emotion: 0 for emotion in emotions}
    
    # Generate systematic combinations first to ensure diversity
    for field in FIELDS:
        topics = TOPICS[field]
        for emotion in emotions:
            templates = TEMPLATES[emotion]
            for topic in topics:
                for template in templates[:10]: # Use first 10 templates for system combos
                    query = template.format(topic=topic)
                    dataset.append({
                        "text": query,
                        "emotion": emotion,
                        "field": field
                    })
                    counts[emotion] += 1

    # Fill up the rest with random choices until we reach target_count
    while len(dataset) < target_count:
        emotion = random.choice(emotions)
        field = random.choice(FIELDS)
        topic = random.choice(TOPICS[field])
        template = random.choice(TEMPLATES[emotion])
        
        query = template.format(topic=topic)
        # Add slight variations (casing, trailing punctuation)
        if random.random() > 0.8:
            query = query.lower()
        if random.random() > 0.8:
            if query.endswith('.'):
                query = query[:-1]
        
        dataset.append({
            "text": query,
            "emotion": emotion,
            "field": field
        })
        counts[emotion] += 1

    print(f"Generated dataset of size {len(dataset)} items.")
    print("Class distribution:")
    for emotion, count in counts.items():
        print(f"  {emotion}: {count}")
        
    return dataset

def main():
    # Make sure target directory exists
    data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_dir / "student_emotions_dataset.csv"
    
    print("Generating student emotion datasets...")
    data = generate_queries()
    
    # Shuffle dataset
    random.shuffle(data)
    
    # Write to CSV
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotion", "field"])
        for row in data:
            writer.writerow([row["text"], row["emotion"], row["field"]])
            
    print(f"Dataset successfully saved to: {dataset_path}")

if __name__ == "__main__":
    main()
