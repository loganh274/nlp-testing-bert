"""Generate targeted synthetic training examples for weak sentiment classes."""

import csv
import random
from typing import List, Tuple
from datetime import datetime

OUTPUT_PATH = "data/synthetic_training_examples.csv"

# Targeted templates based on the specific confusion patterns identified:
# 1. Negative → Neutral: Professional complaints being missed
# 2. Very Negative → Negative: Intensity not being detected

# VERY NEGATIVE: Strong emotional intensity markers
# These should NOT be confused with regular Negative
VERY_NEGATIVE_TEMPLATES = [
    # Frustration + time (the "years of waiting" pattern)
    "I can't believe this still isn't fixed after all these years. Absolutely unacceptable.",
    "It's been YEARS and still nothing. This is ridiculous.",
    "How many more years do we have to wait? This is insulting.",
    "We've been begging for this for years. Completely ignored.",
    
    # Strong emotional words (appalling, ridiculous, infuriating)
    "I find it appalling that this basic feature doesn't exist.",
    "This is absolutely infuriating. How is this not a priority?",
    "Completely unacceptable for software at this price point.",
    "This is embarrassing. Even free tools have this functionality.",
    
    # Threats and ultimatums
    "Fix this or we're cancelling. Final warning.",
    "We're switching to the competition if this isn't addressed immediately.",
    "This is our dealbreaker. We cannot continue without this.",
    "I regret choosing this software. Considering alternatives now.",
    
    # Direct criticism / personal frustration
    "What a waste of money. This is useless without basic features.",
    "Whoever makes these decisions clearly doesn't use the product.",
    "Stop ignoring your paying customers! This is critical!",
    "I'm completely fed up. This has cost us real business.",
]

# NEGATIVE: Professional disappointment (NOT neutral observations)
# These should NOT be confused with Neutral feature requests
NEGATIVE_TEMPLATES = [
    # "Surprised/disappointed" pattern (from your misclassifications)
    "Really surprised this isn't supported. It's a basic function.",
    "Disappointed that this still isn't available. We need this.",
    "I'm frustrated that this hasn't been prioritized.",
    "Quite disappointing to see this missing. It's standard elsewhere.",
    
    # "Should be basic" pattern (from your misclassifications)  
    "This should be a basic feature. Other platforms handle it easily.",
    "Such a basic function and it's not supported. Frustrating.",
    "This is pretty basic functionality that we expected to have.",
    "Can't believe this isn't standard. It's causing us issues.",
    
    # Business impact (distinguishes from neutral observations)
    "Not having this is causing significant problems for our team.",
    "This limitation is affecting our daily operations.",
    "We're struggling without this feature. It's holding us back.",
    "This is costing us time every single day.",
    
    # Urgent requests (more than just asking)
    "We urgently need this. Please prioritize.",
    "This is becoming a real problem. Any updates?",
    "Please, we really need this addressed soon.",
    "This has been an issue for too long. When will it be fixed?",
]

# Components for template variation
FEATURES = [
    "multiple delivery addresses",
    "bulk editing",
    "custom reporting", 
    "multi-currency support",
    "batch processing",
    "inventory tracking",
    "API integration",
    "mobile functionality",
    "export options",
    "search functionality",
]


def add_feature_context(text: str) -> str:
    """Optionally prepend or append feature context."""
    if random.random() < 0.4:
        feature = random.choice(FEATURES)
        if random.random() < 0.5:
            return f"Regarding {feature}: {text}"
        else:
            return f"{text} We need {feature}."
    return text


def add_variation(text: str) -> str:
    """Add slight variations to avoid exact duplicates."""
    # Prefix variations
    prefixes = ["", "", "", "Honestly, ", "Look, ", "As a paying customer, "]
    
    # Suffix variations  
    suffixes = ["", "", "", " Please address this.", " This is important."]
    
    result = random.choice(prefixes) + text + random.choice(suffixes)
    
    # Occasional emphasis
    if random.random() < 0.1:
        result = result.replace("!", "!!")
    
    return result


def generate_training_data() -> List[Tuple[str, int]]:
    """Generate targeted synthetic training examples."""
    
    examples = []
    
    # Very Negative (0) - 20 examples
    print("Generating 20 Very Negative examples...")
    for template in VERY_NEGATIVE_TEMPLATES:
        base = add_feature_context(template)
        examples.append((base, 0))
        # One variation each
        examples.append((add_variation(add_feature_context(template)), 0))
    
    # Trim to exactly 20
    very_neg = [(t, l) for t, l in examples if l == 0][:20]
    
    # Negative (1) - 20 examples  
    print("Generating 20 Negative examples...")
    negative_examples = []
    for template in NEGATIVE_TEMPLATES:
        base = add_feature_context(template)
        negative_examples.append((base, 1))
        # One variation each
        negative_examples.append((add_variation(add_feature_context(template)), 1))
    
    # Trim to exactly 20
    neg = negative_examples[:20]
    
    all_examples = very_neg + neg
    random.shuffle(all_examples)
    
    return all_examples


def save_to_csv(examples: List[Tuple[str, int]], output_path: str):
    """Save examples to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, label in examples:
            writer.writerow([text, label])
    
    print(f"✅ Saved {len(examples)} examples to {output_path}")


def print_examples(examples: List[Tuple[str, int]]):
    """Print all examples for review."""
    
    print("\n" + "="*60)
    print("VERY NEGATIVE (0) - Should show strong frustration/anger")
    print("="*60)
    for text, label in examples:
        if label == 0:
            print(f"  • \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    
    print("\n" + "="*60)
    print("NEGATIVE (1) - Professional but clearly disappointed/frustrated")
    print("="*60)
    for text, label in examples:
        if label == 1:
            print(f"  • \"{text[:80]}{'...' if len(text) > 80 else ''}\"")


def main():
    print("="*60)
    print("TARGETED SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"Generated at: {datetime.now().isoformat()}")
    print("\nGenerating minimal, targeted examples for weak classes only.")
    print("Focus: Distinguishing Negative from Neutral, Very Negative from Negative\n")
    
    examples = generate_training_data()
    
    print_examples(examples)
    
    save_to_csv(examples, OUTPUT_PATH)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Very Negative (0): {sum(1 for e in examples if e[1] == 0)}")
    print(f"  Negative (1):      {sum(1 for e in examples if e[1] == 1)}")
    print(f"  Total:             {len(examples)}")
    
    print("\n📋 NEXT STEPS:")
    print("  1. Review examples above - remove any that don't fit")
    print("  2. Run: python scripts/merge_training_data.py")
    print("  3. Retrain: python scripts/sentiment_training.py")


if __name__ == "__main__":
    main()