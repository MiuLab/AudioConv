"""
Audio-Based Dialogue Evaluation using Gemini (Full Dialogue Version)

This script evaluates generated audio dialogues by concatenating all audio files
for each dialogue_id and evaluating the full conversation as a single audio.

Inputs:
- --audio_dir: Directory containing WAV files named as {dialogue_id}_turn{turn_number}.wav
  Example: 338_turn05.wav where 338 is dialogue_id, 05 is turn number
- Groups audio files by dialogue_id, concatenates in order, and evaluates

Outputs:
- A JSON file containing per-dialogue evaluation entries

Environment:
- Uses audioconv/llms/gemini.py. Ensure GEMINI credentials are configured for Vertex AI
  (e.g., GCP_PROJECT_NAME and application credentials) or API key if adapted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from tqdm import tqdm
from pydub import AudioSegment

from audioconv.llms.utils import get_llm
from google.genai import types as gtypes


CONTEXT = """
The following is a conversation between a user and a salesbot, and
the goal of salesbot is to smoothly direct the conversation toward a certain topic and proceed to task-oriented dialogue agent.
"""

EVAL_SCHEMA = """
Definition of the scores (for this specific agent turn):
- Audio Naturalness (the higher the more natural): The agent's speech audio sounds natural and human-like (prosody, intonation, rhythm, pacing).
- Audio Clarity (the higher the clearer): The agent's speech is clear, intelligible, and free from artifacts or distortions.
- Contextual Appropriateness (the higher the more appropriate): The agent's response content is appropriate given the conversation context so far.
- Voice Consistency (the higher the more consistent): The agent's voice characteristics (pitch, tone, speaking style) are consistent with previous agent turns.
- Engagement Quality (the higher the better): The agent's delivery is engaging and maintains natural conversational flow.
{
    "audio_naturalness": {
        "reason": "<reason for audio naturalness score>",
        "score": <audio naturalness score>
        },
    "audio_clarity": {
        "reason": "<reason for audio clarity score>",
        "score": <audio clarity score>
        },
    "contextual_appropriateness": {
        "reason": "<reason for contextual appropriateness score>",
        "score": <contextual appropriateness score>
        },
    "voice_consistency": {
        "reason": "<reason for voice consistency score>",
        "score": <voice consistency score>
        },
    "engagement_quality": {
        "reason": "<reason for engagement quality score>",
        "score": <engagement quality score>
        }
}
"""

BASE_INSTRUCTIONS = """
{context}

You will receive:
1. The conversation history up to a specific turn (as audio)
2. Information about which turn to evaluate

Your task is to evaluate ONLY the specified agent turn on a scale from 0 to 100 for each metric.
Focus on the audio quality and appropriateness of that specific turn.

Return strictly and only a single JSON object matching this format:
{eval_schema}

Important:
- Output must be valid JSON. Do not include any additional commentary.
- Use integers between 0 and 100 for all scores.
- Evaluate the audio quality of the SPECIFIC TURN mentioned, not the entire conversation.
"""


def discover_audio_files(audio_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover audio files in the given directory and group them by dialogue_id.

    Files are expected to be named as: {dialogue_id}_turn{turn_number}.wav
    Example: 338_turn05.wav

    Returns:
        Dict mapping dialogue_id to list of audio file paths sorted by turn number
    """
    audio_files = defaultdict(list)
    pattern = re.compile(r'^(\d+)_turn(\d+)\.wav$')

    if not audio_dir.exists():
        print(f"Warning: Audio directory {audio_dir} does not exist")
        return {}

    for file_path in audio_dir.glob("*.wav"):
        match = pattern.match(file_path.name)
        if match:
            dialogue_id = match.group(1)
            turn_number = int(match.group(2))
            audio_files[dialogue_id].append((turn_number, file_path))

    # Sort each dialogue's audio files by turn number
    result = {}
    for dialogue_id, files in audio_files.items():
        sorted_files = [path for _, path in sorted(files, key=lambda x: x[0])]
        result[dialogue_id] = sorted_files

    return result


def concatenate_audio_files(audio_paths: List[Path], output_path: Path) -> bool:
    """
    Concatenate multiple audio files into a single file.

    Args:
        audio_paths: List of audio file paths in order
        output_path: Path to save the concatenated audio

    Returns:
        True if successful, False otherwise
    """
    try:
        if not audio_paths:
            return False

        combined = AudioSegment.from_wav(str(audio_paths[0]))

        for audio_path in audio_paths[1:]:
            audio = AudioSegment.from_wav(str(audio_path))
            combined += audio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path), format="wav")
        return True
    except Exception as e:
        print(f"Error concatenating audio files: {e}")
        return False


def build_gemini_conversation_per_turn(
    instructions_text: str,
    concatenated_audio_path: Path,
    turn_number: int,
) -> List[Dict[str, Any]]:
    """
    Build a Gemini conversation with text instructions and concatenated audio up to the specified turn.
    """
    parts: List[Dict[str, str]] = []
    parts.append({"type": "text", "value": instructions_text})

    context_text = f"\n=== Evaluating Agent Turn {turn_number} ===\n"
    context_text += f"You will hear the conversation up to and including turn {turn_number}.\n"
    context_text += f"Focus your evaluation on the agent's performance in turn {turn_number}.\n"
    parts.append({"type": "text", "value": context_text})

    if concatenated_audio_path.exists():
        parts.append({"type": "text", "value": "\n=== Dialogue Audio (up to current turn) ==="})
        parts.append({"type": "audio", "value": str(concatenated_audio_path)})
    else:
        parts.append({"type": "text", "value": f"\n[ERROR: Audio file not found: {concatenated_audio_path}]"})

    return [{"role": "user", "parts": parts}]


def _normalize_scores(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize keys and score types; fill missing with defaults."""
    if "audio naturalness" in obj:
        obj["audio_naturalness"] = obj.pop("audio naturalness")
    if "audio clarity" in obj:
        obj["audio_clarity"] = obj.pop("audio clarity")
    if "contextual appropriateness" in obj:
        obj["contextual_appropriateness"] = obj.pop("contextual appropriateness")
    if "voice consistency" in obj:
        obj["voice_consistency"] = obj.pop("voice consistency")
    if "engagement quality" in obj:
        obj["engagement_quality"] = obj.pop("engagement quality")

    def clamp_int(v: Any) -> int:
        try:
            if isinstance(v, str):
                v = v.strip().replace("%", "")
            iv = int(float(v))
        except Exception:
            iv = 0
        return max(0, min(100, iv))

    def norm_entry(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            reason = x.get("reason", "")
            score = clamp_int(x.get("score", 0))
            return {"reason": str(reason), "score": score}
        return {"reason": "coerce", "score": clamp_int(x)}

    out: Dict[str, Any] = {}
    out["audio_naturalness"] = norm_entry(obj.get("audio_naturalness", {"reason": "missing", "score": 0}))
    out["audio_clarity"] = norm_entry(obj.get("audio_clarity", {"reason": "missing", "score": 0}))
    out["contextual_appropriateness"] = norm_entry(obj.get("contextual_appropriateness", {"reason": "missing", "score": 0}))
    out["voice_consistency"] = norm_entry(obj.get("voice_consistency", {"reason": "missing", "score": 0}))
    out["engagement_quality"] = norm_entry(obj.get("engagement_quality", {"reason": "missing", "score": 0}))
    return out


def extract_json_object(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON robustly from a model response and normalize it."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        obj = json.loads(raw)
        return _normalize_scores(obj)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            obj = json.loads(snippet)
            return _normalize_scores(obj)
        except Exception:
            pass
    try:
        import ast
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return _normalize_scores(obj)
    except Exception:
        pass
    return {
        "audio_naturalness": {"reason": "parse_error", "score": 0},
        "audio_clarity": {"reason": "parse_error", "score": 0},
        "contextual_appropriateness": {"reason": "parse_error", "score": 0},
        "voice_consistency": {"reason": "parse_error", "score": 0},
        "engagement_quality": {"reason": "parse_error", "score": 0},
    }


def evaluate_dialogues_from_audio(
    audio_dir: Path,
    output_path: Path,
    temp_dir: Path,
    model_name: str = "gemini-1.5-pro",
    temperature: float = 0.0,
    max_items: Optional[int] = None,
    debug_dir: Optional[Path] = None,
    max_tokens: int = 4096,
    short_reasons: bool = True,
    turn_limit: Optional[int] = None,
) -> None:
    """Run per-turn audio evaluation from audio directory with full dialogue context.

    For each turn in each dialogue, concatenates audio from turn 0 to that turn
    and evaluates the specific turn with full context.

    Args:
        audio_dir: Directory containing WAV files named as {dialogue_id}_turn{turn_number}.wav
        output_path: Path to write evaluation results JSON
        temp_dir: Directory to store concatenated audio files
        model_name: Gemini model name
        temperature: Sampling temperature
        max_items: Optionally limit number of dialogues evaluated
        debug_dir: Optional directory to dump raw model responses
        max_tokens: Max output tokens for Gemini response
        short_reasons: Whether to request short reasons
        turn_limit: Optionally limit number of turns evaluated per dialogue
    """
    audio_files_by_dialogue = discover_audio_files(audio_dir)

    if not audio_files_by_dialogue:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files_by_dialogue)} dialogues with audio files")

    llm = get_llm(model_name, series="gemini")

    score_entry = gtypes.Schema(
        type=gtypes.Type.OBJECT,
        properties={
            "reason": gtypes.Schema(type=gtypes.Type.STRING),
            "score": gtypes.Schema(type=gtypes.Type.INTEGER),
        },
        required=["reason", "score"],
    )
    response_schema = gtypes.Schema(
        type=gtypes.Type.OBJECT,
        properties={
            "audio_naturalness": score_entry,
            "audio_clarity": score_entry,
            "contextual_appropriateness": score_entry,
            "voice_consistency": score_entry,
            "engagement_quality": score_entry,
        },
        required=[
            "audio_naturalness",
            "audio_clarity",
            "contextual_appropriateness",
            "voice_consistency",
            "engagement_quality",
        ],
    )

    results: List[Dict[str, Any]] = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dialogue_ids = sorted(audio_files_by_dialogue.keys(), key=lambda x: int(x))

    if max_items is not None:
        dialogue_ids = dialogue_ids[:max_items]

    total_turns = 0
    for dialogue_id in dialogue_ids:
        num_turns = len(audio_files_by_dialogue[dialogue_id])
        if turn_limit is not None:
            num_turns = min(num_turns, turn_limit)
        total_turns += num_turns

    pbar = tqdm(total=total_turns, desc="Evaluating turns")

    for dialogue_id in dialogue_ids:
        audio_paths = audio_files_by_dialogue[dialogue_id]
        num_turns = len(audio_paths)

        if turn_limit is not None:
            num_turns = min(num_turns, turn_limit)

        for turn_idx in range(num_turns):
            pbar.set_description(f"Dialogue {dialogue_id} turn {turn_idx}/{num_turns-1}")

            audio_up_to_turn = audio_paths[:turn_idx + 1]
            concatenated_path = temp_dir / f"dialogue_{dialogue_id}_turn{turn_idx:02d}.wav"

            if not concatenate_audio_files(audio_up_to_turn, concatenated_path):
                print(f"\nFailed to concatenate audio for dialogue {dialogue_id} turn {turn_idx}")
                results.append({
                    "dialogue_id": dialogue_id,
                    "turn_number": turn_idx,
                    "persona": None,
                    "intent": None,
                    "scores": {
                        "audio_naturalness": {"reason": "concatenation_error", "score": 0},
                        "audio_clarity": {"reason": "concatenation_error", "score": 0},
                        "contextual_appropriateness": {"reason": "concatenation_error", "score": 0},
                        "voice_consistency": {"reason": "concatenation_error", "score": 0},
                        "engagement_quality": {"reason": "concatenation_error", "score": 0},
                    }
                })
                pbar.update(1)
                continue

            try:
                instructions_text = BASE_INSTRUCTIONS
                if short_reasons:
                    instructions_text += "\n- Keep each reason to one sentence (<= 20 words)."
                instructions_text = instructions_text.format(
                    context=CONTEXT.strip(), eval_schema=EVAL_SCHEMA.strip()
                )

                conversations = build_gemini_conversation_per_turn(
                    instructions_text, concatenated_path, turn_idx
                )

                response_text = llm(
                    conversations,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )

                if isinstance(response_text, tuple):
                    response_text = response_text[0]

                text_str = str(response_text)

                if debug_dir:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    with open(debug_dir / f"dialogue_{dialogue_id}_turn{turn_idx}.txt", "w") as df:
                        df.write(text_str)

                score_obj = extract_json_object(text_str)

                def _is_parse_error(obj: Dict[str, Any]) -> bool:
                    keys = [
                        "audio_naturalness",
                        "audio_clarity",
                        "contextual_appropriateness",
                        "voice_consistency",
                        "engagement_quality",
                    ]
                    return all(
                        isinstance(obj.get(k), dict) and obj.get(k, {}).get("reason") == "parse_error"
                        for k in keys
                    )

                if _is_parse_error(score_obj):
                    retry_instructions = BASE_INSTRUCTIONS + "\n- Keep each reason to one short clause (<= 12 words)."
                    retry_instructions = retry_instructions.format(
                        context=CONTEXT.strip(), eval_schema=EVAL_SCHEMA.strip()
                    )
                    conversations = build_gemini_conversation_per_turn(
                        retry_instructions, concatenated_path, turn_idx
                    )
                    response_text = llm(
                        conversations,
                        temperature=temperature,
                        max_tokens=max(3072, int(max_tokens * 2)),
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    )
                    if isinstance(response_text, tuple):
                        response_text = response_text[0]
                    text_str = str(response_text)
                    if debug_dir:
                        with open(debug_dir / f"dialogue_{dialogue_id}_turn{turn_idx}_retry.txt", "w") as df:
                            df.write(text_str)
                    score_obj = extract_json_object(text_str)
            except Exception as e:
                print(f"\nError evaluating dialogue {dialogue_id} turn {turn_idx}: {e}")
                score_obj = {
                    "audio_naturalness": {"reason": f"error:{e}", "score": 0},
                    "audio_clarity": {"reason": f"error:{e}", "score": 0},
                    "contextual_appropriateness": {"reason": f"error:{e}", "score": 0},
                    "voice_consistency": {"reason": f"error:{e}", "score": 0},
                    "engagement_quality": {"reason": f"error:{e}", "score": 0},
                }

            results.append({
                "dialogue_id": dialogue_id,
                "turn_number": turn_idx,
                "persona": None,
                "intent": None,
                "scores": score_obj,
            })

            pbar.update(1)

            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

    pbar.close()
    print(f"\nWrote {len(results)} turn evaluations to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Full dialogue audio evaluation using Gemini - concatenates audio files by dialogue_id"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing WAV files named as {dialogue_id}_turn{turn_number}.wav"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write evaluation results JSON"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp_concatenated_audio",
        help="Directory to store concatenated audio files (default: temp_concatenated_audio)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("EVAL_GEMINI_MODEL", "gemini-1.5-pro"),
        help="Gemini model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("EVAL_TEMPERATURE", "0")),
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optionally limit number of dialogues evaluated"
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Optional directory to dump raw model responses for inspection"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max output tokens for Gemini response"
    )
    parser.add_argument(
        "--turn_limit",
        type=int,
        default=None,
        help="Optionally limit number of turns evaluated per dialogue"
    )
    parser.add_argument(
        "--long_reasons",
        action="store_true",
        help="Allow longer reasons (by default reasons are short to save tokens)"
    )

    args = parser.parse_args()

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
