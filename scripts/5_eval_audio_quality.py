"""
Step 5: Evaluate Audio Quality (Gemini Perceptual Evaluator)

Evaluates generated agent audio files using Gemini as a perceptual judge.
For each turn in each dialogue, concatenates audio up to that turn and
scores on: audio naturalness, clarity, contextual appropriateness,
voice consistency, and engagement quality (0-100 each).

Requires:
- GEMINI_API_KEY environment variable
- GCP_PROJECT_NAME environment variable

Usage:
  python scripts/5_eval_audio_quality.py \
    --audio_dir dialogue_outputs/agent_sesame_gpt-4o-mini_speaker1_fixed_eval \
    --output eval_results/audio_quality_sesame.json \
    --model gemini-2.5-flash

  # Dry run (discover files without calling API)
  python scripts/5_eval_audio_quality.py \
    --audio_dir dialogue_outputs/agent_mimo-audio_fixed_eval \
    --output eval_results/audio_quality_mimo.json \
    --dry_run
"""

import sys
import os
import argparse
from pathlib import Path

# Allow running as a script from the AudioConv root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.audio_quality import evaluate_dialogues_from_audio, discover_audio_files, concatenate_audio_files


def main():
    parser = argparse.ArgumentParser(
        description="Audio quality evaluation using Gemini perceptual evaluator"
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Directory containing WAV files named {dialogue_id}_turn{N}.wav")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to write evaluation results JSON")
    parser.add_argument("--temp_dir", type=str, default="temp_concatenated_audio",
                       help="Temp directory for concatenated audio files")
    parser.add_argument("--model", type=str,
                       default=os.getenv("EVAL_GEMINI_MODEL", "gemini-2.5-flash"),
                       help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_items", type=int, default=None,
                       help="Optionally limit number of dialogues evaluated")
    parser.add_argument("--debug_dir", type=str, default=None,
                       help="Optional directory to dump raw model responses")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Max output tokens for Gemini response")
    parser.add_argument("--turn_limit", type=int, default=None,
                       help="Optionally limit number of turns per dialogue")
    parser.add_argument("--long_reasons", action="store_true",
                       help="Allow longer reasons (default: short to save tokens)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Discover files and test concatenation without calling API")

    args = parser.parse_args()

    if args.dry_run:
        print(f"DRY RUN MODE - Discovering audio files from {args.audio_dir}")
        audio_files_by_dialogue = discover_audio_files(Path(args.audio_dir))

        if not audio_files_by_dialogue:
            print(f"No audio files found in {args.audio_dir}")
            return

        print(f"\nFound {len(audio_files_by_dialogue)} dialogues with audio files")
        dialogue_ids = sorted(audio_files_by_dialogue.keys(), key=lambda x: int(x))
        print(f"\nSample dialogue IDs (first 10): {dialogue_ids[:10]}")

        first_id = dialogue_ids[0]
        first_files = audio_files_by_dialogue[first_id]
        print(f"\nDialogue {first_id} has {len(first_files)} audio files:")
        for i, path in enumerate(first_files[:5]):
            print(f"  Turn {i}: {path.name}")
        if len(first_files) > 5:
            print(f"  ... and {len(first_files) - 5} more")

        print(f"\nTesting audio concatenation for dialogue {first_id}...")
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_output = temp_dir / f"test_dialogue_{first_id}_full.wav"

        if concatenate_audio_files(first_files, test_output):
            print(f"✓ Successfully concatenated audio to: {test_output}")
            import soundfile as sf
            data, samplerate = sf.read(test_output)
            duration = len(data) / samplerate
            print(f"  Duration: {duration:.2f} seconds")
        else:
            print("✗ Failed to concatenate audio")

        print(f"\nDry run complete. Remove --dry_run to run actual evaluation.")
        return

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please run: export GEMINI_API_KEY='your-key-here'")
        return
    if not os.getenv("GCP_PROJECT_NAME"):
        print("ERROR: GCP_PROJECT_NAME environment variable not set!")
        print("Please run: export GCP_PROJECT_NAME='your-gcp-project'")
        return

    evaluate_dialogues_from_audio(
        audio_dir=Path(args.audio_dir),
        output_path=Path(args.output),
        temp_dir=Path(args.temp_dir),
        model_name=args.model,
        temperature=args.temperature,
        max_items=args.max_items,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        max_tokens=args.max_tokens,
        short_reasons=not args.long_reasons,
        turn_limit=args.turn_limit,
    )


if __name__ == "__main__":
    main()
