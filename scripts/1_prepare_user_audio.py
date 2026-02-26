"""
Step 1: Prepare User Audio from MSGD Dataset

Loads MSGD_dataset_final.json, samples dialogues by intent distribution,
and generates user-side WAV files using the UserSimulator (GPT + Sesame CSM TTS).

Output: data/user_audio/merge_XXXX_turnYY.wav
  - XXXX = zero-padded dialogue ID (e.g., 0042)
  - YY   = zero-padded turn index (e.g., 00, 02, 04 for even = user turns)

Usage:
  python scripts/1_prepare_user_audio.py \
    --dataset data/MSGD_dataset_final.json \
    --output_dir data/user_audio \
    --num_dialogues 500 \
    --voice_profile_dir data/voice_profiles/user \
    --openai_model gpt-4o-mini
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from audioconv.simulators.user_simulator import UserSimulator


def load_msgd_dataset(dataset_path: str) -> Dict[str, Dict]:
    """Load MSGD dataset and organize by dialogue ID."""
    print(f"Loading MSGD dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    dialogues = {}
    for item in dataset:
        dialogue_id = item["id"].replace("merge_", "")
        dialogues[dialogue_id] = item

    print(f"✓ Loaded {len(dialogues)} dialogues")
    return dialogues


def sample_by_intent(dialogues: Dict[str, Dict], num_samples: int) -> List[str]:
    """
    Sample dialogue IDs proportionally by intent type.

    Args:
        dialogues: Dict mapping dialogue_id to dialogue data
        num_samples: Total number of dialogues to sample

    Returns:
        List of sampled dialogue IDs
    """
    # Group by intent
    by_intent: Dict[str, List[str]] = defaultdict(list)
    for did, data in dialogues.items():
        intent = data.get("intent", {}).get("type", "Unknown")
        by_intent[intent].append(did)

    intents = list(by_intent.keys())
    n_intents = len(intents)
    per_intent = num_samples // n_intents
    remainder = num_samples % n_intents

    sampled = []
    for i, intent in enumerate(intents):
        n = per_intent + (1 if i < remainder else 0)
        candidates = sorted(by_intent[intent])
        sampled.extend(candidates[:n])

    # Sort for reproducibility
    return sorted(sampled[:num_samples])


def get_user_turns(dialogue_data: Dict) -> List[Dict]:
    """
    Extract user turns from a dialogue.

    Returns:
        List of dicts with 'text' and 'turn_index'
    """
    dialog_turns = dialogue_data.get("dialog", [])
    user_turns = []
    for i, turn in enumerate(dialog_turns):
        if turn.startswith("User:"):
            text = turn.replace("User:", "").strip()
            user_turns.append({
                "text": text,
                "turn_index": i,
            })
    return user_turns


def main():
    parser = argparse.ArgumentParser(
        description="Generate user audio files from MSGD dataset using Sesame CSM TTS"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/MSGD_dataset_final.json",
        help="Path to MSGD_dataset_final.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/user_audio",
        help="Output directory for user audio WAV files"
    )
    parser.add_argument(
        "--num_dialogues",
        type=int,
        default=500,
        help="Number of dialogues to sample (default: 500)"
    )
    parser.add_argument(
        "--dialogue_ids",
        type=str,
        nargs="*",
        default=None,
        help="Specific dialogue IDs to process (overrides --num_dialogues)"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for user text generation"
    )
    parser.add_argument(
        "--csm_model",
        type=str,
        default="sesame/csm-1b",
        help="Sesame CSM model ID"
    )
    parser.add_argument(
        "--voice_profile_dir",
        type=str,
        default=None,
        help="Directory with voice profile audio samples for voice cloning (optional)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip already-generated audio files (default: True)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Regenerate all audio files even if they exist"
    )

    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return

    # Load dataset
    dialogues = load_msgd_dataset(args.dataset)

    # Determine which dialogues to process
    if args.dialogue_ids:
        dialogue_ids = args.dialogue_ids
        print(f"Processing {len(dialogue_ids)} specified dialogues")
    else:
        dialogue_ids = sample_by_intent(dialogues, args.num_dialogues)
        print(f"Sampled {len(dialogue_ids)} dialogues by intent distribution")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize UserSimulator
    print(f"\nInitializing UserSimulator with model: {args.csm_model}")
    simulator = UserSimulator(
        model_id=args.csm_model,
        openai_model=args.openai_model,
        offload_to_cpu=True,
    )

    # Load voice profile if provided
    if args.voice_profile_dir:
        vp_dir = Path(args.voice_profile_dir)
        audio_files = sorted(vp_dir.glob("*.wav"))
        if audio_files:
            # Simple: use filenames (without extension) as text placeholders
            texts = [f.stem.replace("_", " ") for f in audio_files]
            simulator.load_voice_profile(
                audio_files=[str(f) for f in audio_files],
                texts=texts,
                profile_name=vp_dir.name
            )
            print(f"✓ Loaded voice profile from {vp_dir}")

    # Generate audio for each dialogue
    total_generated = 0
    total_skipped = 0

    print(f"\nGenerating user audio to {output_dir}")
    print(f"{'='*60}\n")

    for dialogue_id in dialogue_ids:
        if dialogue_id not in dialogues:
            print(f"Warning: Dialogue {dialogue_id} not found in dataset, skipping")
            continue

        dialogue_data = dialogues[dialogue_id]
        user_turns = get_user_turns(dialogue_data)

        if not user_turns:
            print(f"Warning: No user turns found for dialogue {dialogue_id}, skipping")
            continue

        print(f"Dialogue {dialogue_id}: {len(user_turns)} user turns")

        for turn_info in user_turns:
            turn_idx = turn_info["turn_index"]
            user_text = turn_info["text"]

            # Output filename: merge_XXXX_turnYY.wav
            out_filename = f"merge_{dialogue_id.zfill(4)}_turn{turn_idx:02d}.wav"
            out_path = output_dir / out_filename

            # Skip if already exists
            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            print(f"  Turn {turn_idx:02d}: {user_text[:60]}...")

            try:
                # Convert user text to speech
                simulator.text_to_speech(user_text, str(out_path))
                total_generated += 1
                print(f"  ✓ Saved: {out_filename}")
            except Exception as e:
                print(f"  ✗ Error generating audio for turn {turn_idx}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated: {total_generated} audio files")
    print(f"  Skipped (existing): {total_skipped} audio files")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
