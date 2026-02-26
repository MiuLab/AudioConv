"""
Step 4: Evaluate Dialogue Quality (LLM-as-Judge)

Evaluates agent transcripts using GPT-4 or Gemini as judge.
Scores each dialogue turn on naturalness and consistency (0-100).

Usage:
  # Evaluate audio agent transcripts
  python scripts/4_eval_dialogue_quality.py \
    --transcript_dir dialogue_outputs/agent_sesame_gpt-4o-mini_speaker1_fixed_eval \
    --output_file eval_results/dialogue_quality_sesame.json \
    --model gpt-4o-mini

  # Evaluate text-only agent transcripts with Gemini
  python scripts/4_eval_dialogue_quality.py \
    --transcript_dir dialogue_outputs_text_only/agent_gpt-4o-mini_text_only_fixed_eval \
    --output_file eval_results/dialogue_quality_gpt4omini.json \
    --model gemini-2.5-flash \
    --msgd_dataset data/MSGD_dataset_final.json
"""

import sys
from pathlib import Path

# Allow running as a script from the AudioConv root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dialogue_quality import main

if __name__ == "__main__":
    main()
