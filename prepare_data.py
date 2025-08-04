"""
The prepare_data.py file takes care of loading the structured Strategic Blueprint that was provided in the case study
In this implementation, I leverage insightfyl and  varied instruction–context–response examples
Sequel to this, I write them out as JSONL to training_data.jsonl in my Drive.
"""

import os
import json

# 1. Paths: project root and data directory
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    # 2. Load the Strategic Blueprint as a Python dict
    blueprint = json.loads(r'''
{
  "target_persona": "The 'Constrained Strategist': A marketing leader who projects data-driven confidence externally, but internally experiences constant anxiety and doubt due to fragmented data and uncertain attribution.",
  "core_insight": "The primary pain point for marketers is not the financial cost of wasted ad spend, but the intense, perpetual psychological burden and cognitive tax of making high-stakes decisions under conditions of constant uncertainty.",
  "persuasion_tactic": "Vulnerability-to-Power Transition: Acknowledge the marketer's current state of vulnerability and doubt, then frame the solution as the direct path to reclaiming control, confidence, and strategic power.",
  "key_message": "The goal is 'Mental Liberation.' The solution offers not just better data, but freedom from decision anxiety, allowing a shift from reactive guesswork to confident, unburdened leadership.",
  "call_to_action": "Adopt a new operational framework that transforms data chaos into strategic clarity and decision-making confidence."
}
''')
    # 3. Define a set of varied, persona-driven instructions
    instructions = [
        "Write a compelling LinkedIn post that starts with acknowledging marketers’ decision-anxiety due to fragmented data. Next, introduce our 'Mental Liberation' toolkit as the game-changing solution.",
        "Craft a 30-character PPC headline that calls out 'Data Overwhelm', then promises to unleash new capabilities for marketing leaders with our new operational framework.",
        "Compose a 150-word email that teases how our tool slashes a marketer's anxiety over data by 50%, spearheading exponential increases in clickthrough rates for overburdened CMOs.",
        "Craft an irresistible LinkedIn InMail that starts, 'When was the last time you felt truly confident in your campaign numbers?' then offers a one-click invite to a Mental Liberation masterclass.",
        "Draft a 30-second podcast spot that opens with the host confessing, 'Even I second-guess my campaign data at 2 am,' then transitions to 'Mental Liberation' as the revolutionary answer."
    ]

    # 4. Build examples as dictionaries with instruction, context, and empty response
    examples = []
    for instr in instructions:
        examples.append({
            "instruction": instr,
            "context": blueprint,
            "response": ""  # Placeholder; to be populated by generate_responses.py
        })

    # 5. Write all examples to JSONL in the data directory
    out_path = os.path.join(DATA_DIR, "training_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    main()
