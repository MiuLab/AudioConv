"""
Step 6: Analyze Failure Patterns

Classifies agent dialogue turns into 5 failure categories using LLM-as-judge:
1. Generic responses - Ignoring user's specific context
2. Topic switching - Abruptly changing subject without addressing user
3. Repetition - Repeating same phrases without progression
4. Misunderstanding - Misinterpreting user's clear questions
5. Lack of memory - Forgetting previous turns in the conversation

Usage:
  # Analyze audio agent failures
  python scripts/6_analyze_failures.py \
    --transcript_dir dialogue_outputs/agent_omni_3b-ethan_fixed_eval \
    --output_file eval_results/failure_patterns_omni-3b.json \
    --model gpt-4o-mini

  # With Gemini 2.5 Flash (and MSGD fallback for missing user text)
  python scripts/6_analyze_failures.py \
    --transcript_dir dialogue_outputs/agent_mimo-audio_fixed_eval \
    --msgd_dataset data/MSGD_dataset_final.json \
    --output_file eval_results/failure_patterns_mimo.json \
    --model gemini-2.5-flash
"""

import sys
from pathlib import Path

# Allow running as a script from the AudioConv root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.failure_analysis import main

if __name__ == "__main__":
    main()
