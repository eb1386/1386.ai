#!/usr/bin/env python3
# synthetic instruction generation

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Prompt templates for diverse instruction generation ──────────────

CATEGORY_PROMPTS = {
    "factual_qa": {
        "weight": 0.20,
        "system": "You are a knowledgeable assistant. Give accurate, concise answers. Keep responses under 150 words unless the question requires more detail.",
        "seed_prompts": [
            "What is {topic}?",
            "Explain how {topic} works.",
            "Who was {person} and why are they important?",
            "What is the difference between {thing_a} and {thing_b}?",
            "Where is {place} located and what is it known for?",
            "When did {event} happen and what caused it?",
            "Why is {concept} important in {field}?",
            "What are the main types of {category}?",
        ],
        "fill_vars": {
            "topic": [
                "photosynthesis", "the water cycle", "plate tectonics",
                "the immune system", "inflation in economics", "natural selection",
                "the periodic table", "DNA replication", "the greenhouse effect",
                "supply and demand", "the Pythagorean theorem", "the solar system",
                "the French Revolution", "the Renaissance", "machine learning",
                "quantum mechanics", "the human brain", "cellular respiration",
                "democracy", "the Industrial Revolution", "black holes",
                "the Roman Empire", "the Cold War", "antibiotics",
                "tectonic plates", "the stock market", "the nitrogen cycle",
            ],
            "person": [
                "Albert Einstein", "Marie Curie", "Isaac Newton",
                "Charles Darwin", "Nikola Tesla", "Ada Lovelace",
                "Leonardo da Vinci", "Aristotle", "Galileo",
                "Alexander the Great", "Cleopatra", "Napoleon Bonaparte",
                "Abraham Lincoln", "Martin Luther King Jr.", "Mahatma Gandhi",
            ],
            "thing_a": ["weather", "a virus", "speed", "mass", "an atom", "a republic", "a lake"],
            "thing_b": ["climate", "a bacteria", "velocity", "weight", "a molecule", "a democracy", "a sea"],
            "place": [
                "the Amazon Rainforest", "the Sahara Desert", "the Great Barrier Reef",
                "Mount Everest", "the Mariana Trench", "Antarctica",
                "the Nile River", "the Mediterranean Sea", "Iceland",
            ],
            "event": [
                "the fall of the Berlin Wall", "the moon landing",
                "the discovery of penicillin", "the invention of the printing press",
                "the American Revolution", "World War I",
            ],
            "concept": [
                "biodiversity", "the rule of law", "compound interest",
                "the scientific method", "separation of powers", "vaccination",
            ],
            "field": [
                "biology", "economics", "medicine", "physics",
                "political science", "computer science",
            ],
            "category": [
                "renewable energy", "chemical bonds", "government systems",
                "clouds", "rocks", "economic systems", "literary genres",
            ],
        },
    },
    "reasoning": {
        "weight": 0.20,
        "system": "You are a helpful assistant that thinks step by step. Show your reasoning clearly before giving the final answer. Be concise but thorough.",
        "seed_prompts": [
            "If {scenario}, what would happen and why?",
            "{math_problem}",
            "A {person_a} and a {person_b} are debating about {topic}. What are the strongest arguments on each side?",
            "What are three possible explanations for why {observation}?",
            "Walk me through how to solve this: {problem}",
            "Is the following statement true or false, and why? \"{claim}\"",
        ],
        "fill_vars": {
            "scenario": [
                "the Earth suddenly stopped rotating",
                "all the ice on Earth melted overnight",
                "humans could photosynthesize like plants",
                "gravity was twice as strong",
                "the internet disappeared for a month",
                "a country banned all fossil fuels immediately",
            ],
            "math_problem": [
                "A train travels 120 miles in 2 hours. If it speeds up by 50%, how long would the same trip take?",
                "You have 3 red balls, 4 blue balls, and 5 green balls in a bag. If you draw 2 balls without replacement, what is the probability both are red?",
                "A rectangular garden is 3 times as long as it is wide. If the perimeter is 64 meters, what are the dimensions?",
                "If a shirt costs $40 after a 20% discount, what was the original price?",
                "A population of bacteria doubles every 3 hours. Starting with 100 bacteria, how many will there be after 12 hours?",
                "Two cars start driving toward each other from cities 300 miles apart. One goes 60 mph and the other goes 40 mph. When do they meet?",
                "You invest $1000 at 5% annual compound interest. How much do you have after 10 years?",
            ],
            "person_a": ["scientist", "economist", "historian", "philosopher", "engineer"],
            "person_b": ["politician", "teacher", "journalist", "activist", "business owner"],
            "topic": [
                "whether AI will replace most jobs",
                "whether space exploration is worth the cost",
                "whether nuclear energy is safe",
                "whether social media does more harm than good",
            ],
            "observation": [
                "people tend to procrastinate on important tasks",
                "some animals migrate thousands of miles every year",
                "cities tend to grow near rivers",
                "music affects people's mood",
            ],
            "problem": [
                "finding the area of a triangle with sides 5, 12, and 13",
                "converting 72 degrees Fahrenheit to Celsius",
                "calculating how much paint you need for a room that is 12x15 feet with 8-foot ceilings",
            ],
            "claim": [
                "Lightning never strikes the same place twice",
                "We only use 10% of our brain",
                "The Great Wall of China is visible from space",
                "Water always boils at 100 degrees Celsius",
                "Diamonds are formed from compressed coal",
            ],
        },
    },
    "creative": {
        "weight": 0.10,
        "system": "You are a creative writing assistant. Write vivid, engaging text. Vary your sentence structure and word choice. Keep it natural and avoid cliches.",
        "seed_prompts": [
            "Write a short story (100-200 words) about {story_topic}.",
            "Describe {scene} using vivid sensory details in 2-3 paragraphs.",
            "Write a short poem about {poem_topic}.",
            "Rewrite this sentence to be more vivid and engaging: \"{bland_sentence}\"",
            "Write the opening paragraph of a novel about {novel_premise}.",
        ],
        "fill_vars": {
            "story_topic": [
                "a letter that arrives 50 years late",
                "the last tree on Earth",
                "a clock that runs backwards",
                "a conversation between two strangers on a train",
                "finding something unexpected in an old book",
                "a lighthouse keeper's last night on duty",
                "a robot discovering music for the first time",
            ],
            "scene": [
                "a busy morning market in a small coastal town",
                "a thunderstorm rolling across an open prairie",
                "a quiet library at midnight",
                "the view from a mountain peak at sunrise",
                "an abandoned factory being reclaimed by nature",
                "a crowded subway car during rush hour",
            ],
            "poem_topic": [
                "the changing of seasons", "an old photograph",
                "rain on a tin roof", "the ocean at night",
                "a city waking up", "silence after snowfall",
            ],
            "bland_sentence": [
                "The sunset was nice.",
                "The old house was scary.",
                "She was happy to see her friend.",
                "The food was good.",
                "It was a cold day.",
            ],
            "novel_premise": [
                "a cartographer who discovers her maps are changing on their own",
                "a translator who begins hearing a language no one else can",
                "the last person who remembers what the world was like before",
                "a musician who can only compose in complete darkness",
            ],
        },
    },
    "coding": {
        "weight": 0.15,
        "system": "You are a programming assistant. Write clean, well-commented code. Explain your approach briefly before the code. Use Python unless another language is specified.",
        "seed_prompts": [
            "Write a function that {code_task}.",
            "What does this code do and how could it be improved?\n```python\n{code_snippet}\n```",
            "Explain {programming_concept} with a simple example.",
            "What is the difference between {code_a} and {code_b} in Python?",
            "Write a simple implementation of {algorithm}.",
            "How would you handle {error_scenario} in Python?",
        ],
        "fill_vars": {
            "code_task": [
                "checks if a string is a palindrome",
                "finds the two numbers in a list that add up to a target",
                "reverses a linked list",
                "counts the frequency of each word in a string",
                "flattens a nested list",
                "finds the longest common prefix of a list of strings",
                "checks if two strings are anagrams",
                "implements binary search on a sorted list",
                "merges two sorted lists into one sorted list",
                "removes duplicates from a list while preserving order",
                "finds the nth Fibonacci number using memoization",
                "validates an email address using regex",
                "converts a Roman numeral string to an integer",
            ],
            "code_snippet": [
                "def f(n):\n    return n if n <= 1 else f(n-1) + f(n-2)",
                "x = [i**2 for i in range(10) if i % 2 == 0]",
                "d = {}; [d.update({k: d.get(k, 0) + 1}) for k in words]",
                "result = list(filter(lambda x: x > 0, numbers))",
            ],
            "programming_concept": [
                "recursion", "closures", "generators in Python",
                "the difference between a list and a tuple",
                "how dictionaries work internally",
                "what a decorator does", "async/await",
                "object-oriented inheritance", "the GIL in Python",
            ],
            "code_a": [
                "a list", "== ", "append()", "a class", "args",
                "a shallow copy", "is", "a set", "map()",
            ],
            "code_b": [
                "a tuple", "is ", "extend()", "a dataclass", "kwargs",
                "a deep copy", "==", "a frozenset", "a list comprehension",
            ],
            "algorithm": [
                "a stack using a list",
                "a simple hash table",
                "breadth-first search on a graph",
                "a basic linked list",
                "insertion sort",
                "a queue using two stacks",
            ],
            "error_scenario": [
                "a file that might not exist",
                "a network request that could time out",
                "user input that might not be a valid number",
                "a division that could be by zero",
                "a JSON response that might be malformed",
            ],
        },
    },
    "multi_turn": {
        "weight": 0.20,
        "system": "You are a helpful, conversational assistant. Engage naturally with follow-up questions. Keep responses concise but complete.",
        "seed_prompts": [
            # These are multi-turn: the generation prompt asks for a full conversation
            "Generate a natural 3-turn conversation where a user asks about {topic}, then follows up with a related question, then asks for a specific example.",
            "Generate a 2-turn conversation where a user asks how to {task}, then asks about a common mistake to avoid.",
            "Generate a 3-turn conversation where a user asks about {concept}, then says they're confused about a specific part, then asks for an analogy.",
            "Generate a 4-turn conversation where a user is learning about {subject} and asks progressively deeper questions.",
        ],
        "fill_vars": {
            "topic": [
                "how airplanes fly", "the electoral college",
                "how vaccines work", "compound interest",
                "how search engines rank pages", "the water cycle",
                "how the stock market works", "photosynthesis",
                "how computers store information", "the greenhouse effect",
            ],
            "task": [
                "start investing with a small budget",
                "learn a new programming language",
                "write a good resume",
                "train for a marathon",
                "start a small garden",
                "improve their public speaking",
                "organize a large project",
            ],
            "concept": [
                "machine learning", "blockchain", "quantum computing",
                "game theory", "the theory of relativity",
                "how neural networks learn", "natural selection",
                "the concept of infinity in mathematics",
            ],
            "subject": [
                "Python programming", "European history",
                "organic chemistry", "astronomy",
                "music theory", "economics",
                "probability and statistics",
            ],
        },
    },
    "summarization": {
        "weight": 0.05,
        "system": "You are a summarization assistant. Create clear, accurate summaries that capture the key points. Adjust length to the complexity of the source material.",
        "seed_prompts": [
            "Summarize the key ideas of {topic} in 3-4 sentences.",
            "Give a brief overview of {subject} that a high school student could understand.",
            "What are the 3 most important things to know about {concept}?",
        ],
        "fill_vars": {
            "topic": [
                "the theory of evolution by natural selection",
                "how the internet works",
                "the causes of World War I",
                "the basics of supply and demand",
                "the structure of the US government",
                "how DNA encodes genetic information",
                "the history of the printing press",
                "the scientific method",
            ],
            "subject": [
                "climate change", "the Renaissance",
                "how computers work", "the human immune system",
                "the American Civil War", "the basics of nutrition",
            ],
            "concept": [
                "the greenhouse effect", "compound interest",
                "natural selection", "the Big Bang",
                "photosynthesis", "the water cycle",
            ],
        },
    },
    "instruction_following": {
        "weight": 0.10,
        "system": "You are a precise assistant that follows instructions exactly. Pay close attention to format requirements, constraints, and specific requests.",
        "seed_prompts": [
            "List exactly {number} reasons why {topic}. Number each reason.",
            "Explain {concept} in exactly {sentences} sentences.",
            "Compare {thing_a} and {thing_b} using a table with at least 4 rows.",
            "Describe {subject} without using the letter '{letter}'.",
            "Give a {word_count}-word definition of {term}.",
            "Rewrite the following in {style}: \"{text}\"",
        ],
        "fill_vars": {
            "number": ["3", "5", "7"],
            "topic": [
                "exercise is important", "reading books is beneficial",
                "sleep is essential for health", "learning a second language is valuable",
                "teamwork matters", "critical thinking is important",
            ],
            "concept": [
                "gravity", "inflation", "evolution",
                "electricity", "the water cycle", "democracy",
            ],
            "sentences": ["2", "3", "4", "5"],
            "thing_a": ["cats", "electric cars", "Python", "fiction", "the sun"],
            "thing_b": ["dogs", "gas cars", "JavaScript", "nonfiction", "the moon"],
            "subject": [
                "the ocean", "a thunderstorm", "a city at night",
                "a forest in autumn", "a busy kitchen",
            ],
            "letter": ["e", "a", "s", "t"],
            "word_count": ["10", "15", "20", "25"],
            "term": [
                "photosynthesis", "democracy", "algorithm",
                "ecosystem", "capitalism", "entropy",
            ],
            "style": [
                "formal academic language", "casual conversational tone",
                "a pirate's dialect", "a news headline",
            ],
            "text": [
                "The weather is nice today and I think we should go outside.",
                "Learning new things can be challenging but rewarding.",
                "Technology has changed how we communicate with each other.",
            ],
        },
    },
}


def fill_template(template: str, vars_dict: dict[str, list[str]], rng: random.Random) -> str:
    """Fill a template string with randomly chosen values from vars_dict."""
    result = template
    for key, values in vars_dict.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


def generate_prompt(category: str, config: dict, rng: random.Random) -> tuple[str, str]:
    """Generate a (system, user_prompt) pair for a category."""
    system = config["system"]
    template = rng.choice(config["seed_prompts"])
    user_prompt = fill_template(template, config["fill_vars"], rng)
    return system, user_prompt


def format_as_conversation(response_text: str, user_prompt: str,
                           category: str) -> str:
    """Format the API response into our User/Assistant training format."""
    if category == "multi_turn":
        # The response should already be a multi-turn conversation
        # Try to extract and reformat it
        text = response_text.strip()
        # If the model generated proper User/Assistant format, use as-is
        if "User:" in text and "Assistant:" in text:
            return text
        # Otherwise wrap it
        return f"User: {user_prompt}\nAssistant: {text}"
    else:
        return f"User: {user_prompt}\nAssistant: {response_text.strip()}"


def generate_batch_anthropic(prompts: list[tuple[str, str, str]],
                             model: str = "claude-haiku-4-5-20251001",
                             max_tokens: int = 1024) -> list[tuple[str, str, str]]:
    """Generate responses using the Anthropic API.

    Args:
        prompts: List of (category, system, user_prompt) tuples.
        model: Which Claude model to use. Haiku is cheapest and fast.
        max_tokens: Max response length.

    Returns:
        List of (category, user_prompt, response) tuples.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package required. Install with: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    results = []

    for category, system, user_prompt in prompts:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response_text = message.content[0].text
            results.append((category, user_prompt, response_text))
        except Exception as e:
            print(f"    API error: {e}")
            time.sleep(2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic instruction data using Claude API")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Number of instruction-response pairs to generate")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "raw_1.1" / "synthetic_instruct.jsonl"),
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model to use (haiku is cheapest)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of prompts per batch")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens per response")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Get your key from: https://console.anthropic.com/")
        print("  Then: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count existing samples if resuming
    existing = 0
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing:,} existing samples")

    # Calculate how many samples per category
    category_counts = {}
    for name, config in CATEGORY_PROMPTS.items():
        count = int(args.n_samples * config["weight"])
        category_counts[name] = count

    total_target = sum(category_counts.values())
    remaining = args.n_samples - existing

    print(f"Generating {remaining:,} synthetic instruction samples")
    print(f"  Model: {args.model}")
    print(f"  Output: {output_path}")
    print(f"\n  Category breakdown:")
    for name, count in category_counts.items():
        print(f"    {name:25s} {count:6,} ({CATEGORY_PROMPTS[name]['weight']:.0%})")

    # Generate prompts
    all_prompts = []
    for name, count in category_counts.items():
        config = CATEGORY_PROMPTS[name]
        for _ in range(count):
            system, user_prompt = generate_prompt(name, config, rng)
            all_prompts.append((name, system, user_prompt))

    rng.shuffle(all_prompts)

    # Skip already-generated samples
    all_prompts = all_prompts[existing:]

    if not all_prompts:
        print("All samples already generated!")
        return

    # Generate in batches
    t0 = time.time()
    n_generated = existing
    mode = "a" if args.resume else "w"

    with open(output_path, mode, encoding="utf-8") as out:
        for batch_start in range(0, len(all_prompts), args.batch_size):
            batch = all_prompts[batch_start:batch_start + args.batch_size]

            results = generate_batch_anthropic(batch, model=args.model,
                                               max_tokens=args.max_tokens)

            for category, user_prompt, response in results:
                text = format_as_conversation(response, user_prompt, category)
                record = {
                    "text": text,
                    "source": f"synthetic_{category}",
                    "category": category,
                }
                out.write(json.dumps(record) + "\n")
                n_generated += 1

            # Progress update
            elapsed = time.time() - t0
            rate = (n_generated - existing) / max(1, elapsed) * 3600
            eta = (args.n_samples - n_generated) / max(1, rate) * 3600

            if (batch_start // args.batch_size) % 10 == 0:
                print(f"  {n_generated:,}/{args.n_samples:,} "
                      f"({n_generated / args.n_samples * 100:.1f}%) | "
                      f"{rate:.0f}/hr | "
                      f"ETA: {eta / 60:.0f}m")

            out.flush()

    dt = time.time() - t0
    print(f"\nGenerated {n_generated - existing:,} samples in {dt / 60:.1f} minutes")
    print(f"  Total samples: {n_generated:,}")
    print(f"  Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
