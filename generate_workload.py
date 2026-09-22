import json
import random

def generate_workload():
    queries = []

    # 1. Short factual (100 queries)
    facts = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is the speed of light?",
        "Explain photosynthesis briefly.",
        "How many planets are in the solar system?",
        "What is the square root of 144?",
        "Who was the first president of the United States?",
        "What is the boiling point of water in Celsius?",
        "Translate 'Hello world' to Spanish.",
        "What is the largest mammal on Earth?",
    ]
    for _ in range(100):
        queries.append({
            "type": "short_factual",
            "prompt": random.choice(facts)
        })

    # 2. Multi-turn (50 queries)
    multi_turns = [
        [
            "Can you explain what a binary tree is?",
            "How do I traverse it in-order?",
            "Can you write a Python function for that?",
        ],
        [
            "I want to bake a cake, what ingredients do I need?",
            "How long should I bake it at 350F?",
            "What kind of frosting goes well with it?",
        ],
        [
            "Who won the world cup in 2018?",
            "Who was the captain of that team?",
            "How many goals did they score in the tournament?",
        ]
    ]
    for _ in range(50):
        queries.append({
            "type": "multi_turn",
            "messages": random.choice(multi_turns)
        })

    # 3. Long context (50 queries)
    long_text = " ".join(["This is a very long text intended to test context limits of the language model."] * 200)
    for i in range(50):
        queries.append({
            "type": "long_context",
            "prompt": f"Please summarize the following text: {long_text} (Variant {i})"
        })

    random.shuffle(queries)

    with open("workload.json", "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Generated {len(queries)} queries in workload.json")

if __name__ == "__main__":
    generate_workload()
