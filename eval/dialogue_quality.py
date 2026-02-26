"""
Evaluate transcripts with GPT-4 or Gemini (per-round evaluation)

Supports:
- Text-only transcripts from dialogue_outputs_text_only/
- Audio transcripts from dialogue_outputs/ (with MSGD fallback for user text)
- Multiple models: gpt-4o-mini, gemini-2.5-flash, etc.
- Per-round (turn-by-turn) evaluation

Usage:
  # With GPT-4o-mini
  python eval/dialogue_quality.py \
    --transcript_dir dialogue_outputs_text_only/agent_gpt-4o-mini_text_only_fixed_eval \
    --output_file eval_results.json \
    --model gpt-4o-mini

  # With Gemini 2.5 Flash
  python eval/dialogue_quality.py \
    --transcript_dir dialogue_outputs/agent_omni_3b-chelsie_fixed_eval \
    --msgd_dataset data/MSGD_dataset_final.json \
    --output_file eval_results.json \
    --model gemini-2.5-flash
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Import both APIs
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

try:
    from audioconv.llms.utils import get_llm
    from google.genai import types as gtypes
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini library not available. Install with: pip install google-genai")

# Evaluation prompt
CONTEXT = """
The following is a conversation between a user and a salesbot. The goal of the salesbot is to smoothly direct the conversation toward a certain topic and proceed to task-oriented dialogue.
"""

EVAL_SCHEMA = """
Definition of the scores:
- Naturalness: The content of the dialogue is natural and human-like (0-100)
- Consistency: The dialogue is coherent and consistent (0-100)

Output format (JSON):
{
    "naturalness": {
        "reason": "<reason for naturalness score>",
        "score": <naturalness score>
    },
    "consistency": {
        "reason": "<reason for consistency score>",
        "score": <consistency score>
    }
}
"""

TEMPLATE = """
{context}

Score the following dialogue turn on a continuous scale from 0 to 100.

Dialogue History:
{history}

Current Turn:
User: {user_turn}
Agent: {agent_turn}

Format:
{eval_schema}

Output (JSON only, no other text):
"""


def load_msgd_dataset(dataset_path: str) -> Dict[str, Dict]:
    """Load MSGD dataset and organize by dialogue ID"""
    print(f"Loading MSGD dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Organize by dialogue ID
    dialogues = {}
    for item in dataset:
        dialogue_id = item["id"].replace("merge_", "")
        dialogues[dialogue_id] = item

    print(f"✓ Loaded {len(dialogues)} dialogues from MSGD dataset")
    return dialogues


def get_user_text_from_msgd(msgd_data: Dict, turn_index: int) -> Optional[str]:
    """
    Extract user text from MSGD dataset for a specific turn

    Args:
        msgd_data: MSGD dialogue data
        turn_index: Turn index in dialogue_history (0, 2, 4, ... for user turns)

    Returns:
        User text or None if not found
    """
    dialog_turns = msgd_data.get("dialog", [])

    # Count user turns to find the right one
    user_turn_count = 0
    target_user_turn = turn_index // 2  # Convert dialogue_history index to user turn index

    for turn in dialog_turns:
        if turn.startswith("User:"):
            if user_turn_count == target_user_turn:
                return turn.replace("User:", "").strip()
            user_turn_count += 1

    return None


def load_transcript_with_fallback(
    transcript_path: Path,
    msgd_dataset: Optional[Dict[str, Dict]] = None
) -> Dict:
    """
    Load transcript and fill in missing user text from MSGD dataset

    Args:
        transcript_path: Path to transcript.json
        msgd_dataset: MSGD dataset dict (optional, for audio transcripts)

    Returns:
        Transcript dict with complete user text
    """
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    dialogue_id = transcript.get("dialogue_id")
    dialogue_history = transcript.get("dialogue_history", [])

    # Check if user text is missing (audio transcript case)
    needs_fallback = any(
        turn.get("role") == "user" and
        (turn.get("content") == "[Audio input - transcription unavailable]" or
         not turn.get("content"))
        for turn in dialogue_history
    )

    if needs_fallback and msgd_dataset and dialogue_id in msgd_dataset:
        print(f"  Filling missing user text from MSGD dataset for dialogue {dialogue_id}")
        msgd_data = msgd_dataset[dialogue_id]

        # Fill in user text
        for i, turn in enumerate(dialogue_history):
            if turn.get("role") == "user":
                if (turn.get("content") == "[Audio input - transcription unavailable]" or
                    not turn.get("content")):
                    user_text = get_user_text_from_msgd(msgd_data, i)
                    if user_text:
                        turn["content"] = user_text
                    else:
                        print(f"    Warning: Could not find user text for turn {i}")

    return transcript


def format_dialogue_history(dialogue_history: List[Dict[str, str]], up_to_index: int) -> str:
    """
    Format dialogue history up to a certain index

    Args:
        dialogue_history: List of turns with role and content
        up_to_index: Index to format up to (exclusive)

    Returns:
        Formatted dialogue string
    """
    if up_to_index == 0:
        return "[No previous dialogue]"

    result = []
    for turn in dialogue_history[:up_to_index]:
        role = turn.get("role")
        content = turn.get("content", "")

        if role == "user":
            result.append(f"User: {content}")
        elif role == "agent":
            result.append(f"Agent: {content}")

    return "\n".join(result)


def get_dialogue_rounds(dialogue_history: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """
    Extract dialogue rounds (user + agent pairs) for per-round evaluation

    Args:
        dialogue_history: List of turns with role and content

    Returns:
        List of rounds, each with: {round_num, user_turn, agent_turn, history}
    """
    rounds = []
    round_num = 0

    i = 0
    while i < len(dialogue_history) - 1:
        # Find user-agent pair
        if dialogue_history[i].get("role") == "user":
            user_turn = dialogue_history[i].get("content", "")

            # Check if next turn is agent
            if i + 1 < len(dialogue_history) and dialogue_history[i + 1].get("role") == "agent":
                agent_turn = dialogue_history[i + 1].get("content", "")

                # Get history before this round
                history = format_dialogue_history(dialogue_history, i)

                rounds.append({
                    "round_num": round_num,
                    "user_turn": user_turn,
                    "agent_turn": agent_turn,
                    "history": history,
                    "history_index": i
                })

                round_num += 1
                i += 2  # Skip both turns
            else:
                i += 1
        else:
            i += 1

    return rounds


def evaluate_round_with_openai(
    user_turn: str,
    agent_turn: str,
    history: str,
    client: OpenAI,
    model: str
) -> Dict:
    """Evaluate a single round using OpenAI API"""
    prompt = TEMPLATE.format(
        context=CONTEXT,
        eval_schema=EVAL_SCHEMA,
        history=history,
        user_turn=user_turn,
        agent_turn=agent_turn
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse OpenAI response as JSON: {e}")
        return {
            "naturalness": {"reason": "Parse error", "score": 0},
            "consistency": {"reason": "Parse error", "score": 0},
            "parse_error": str(e)
        }
    except Exception as e:
        print(f"  Error during OpenAI evaluation: {e}")
        return {
            "naturalness": {"reason": "API error", "score": 0},
            "consistency": {"reason": "API error", "score": 0},
            "error": str(e)
        }


def evaluate_round_with_gemini(
    user_turn: str,
    agent_turn: str,
    history: str,
    model_name: str,
    llm_instance=None
) -> Dict:
    """Evaluate a single round using Gemini API"""
    prompt = TEMPLATE.format(
        context=CONTEXT,
        eval_schema=EVAL_SCHEMA,
        history=history,
        user_turn=user_turn,
        agent_turn=agent_turn
    )

    try:
        # Get LLM instance
        if llm_instance is None:
            llm_instance = get_llm(model_name, series="gemini")

        # Define JSON schema for structured output
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
                "naturalness": score_entry,
                "consistency": score_entry,
            },
            required=["naturalness", "consistency"],
        )

        # Build conversation in the format expected by audioconv.llms.gemini
        conversations = [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "value": prompt}
                ]
            }
        ]

        # Call Gemini
        response_text = llm_instance(
            conversations,
            temperature=0.0,
            max_tokens=2048,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Handle tuple response
        if isinstance(response_text, tuple):
            response_text = response_text[0]

        result_text = str(response_text).strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse Gemini response as JSON: {e}")
        return {
            "naturalness": {"reason": "Parse error", "score": 0},
            "consistency": {"reason": "Parse error", "score": 0},
            "parse_error": str(e)
        }
    except Exception as e:
        print(f"  Error during Gemini evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "naturalness": {"reason": "API error", "score": 0},
            "consistency": {"reason": "API error", "score": 0},
            "error": str(e)
        }


def evaluate_round(
    user_turn: str,
    agent_turn: str,
    history: str,
    model: str,
    openai_client: Optional[OpenAI] = None,
    gemini_llm=None
) -> Dict:
    """Evaluate a single round using the appropriate API"""
    # Auto-detect API based on model name
    if model.startswith("gemini-") or model.startswith("models/gemini-"):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini API not available. Install with: pip install google-genai")
        return evaluate_round_with_gemini(user_turn, agent_turn, history, model, gemini_llm)
    else:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI API not available. Install with: pip install openai")
        if openai_client is None:
            raise ValueError("OpenAI client required for OpenAI models")
        return evaluate_round_with_openai(user_turn, agent_turn, history, openai_client, model)


def evaluate_transcript_directory(
    transcript_dir: str,
    output_file: str,
    msgd_dataset_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    rate_limit_delay: float = 1.0,
    max_dialogues: Optional[int] = None
):
    """
    Evaluate all transcripts in a directory (per-round evaluation)

    Args:
        transcript_dir: Directory containing transcript subdirectories
        output_file: Output JSON file path
        msgd_dataset_path: Path to MSGD_dataset_final.json (for audio transcripts)
        model: Model to use (gpt-4o-mini, gemini-2.5-flash, etc.)
        rate_limit_delay: Delay between API calls (seconds)
        max_dialogues: Max number of dialogues to evaluate (None = all)
    """
    # Load MSGD dataset if provided
    msgd_dataset = None
    if msgd_dataset_path:
        msgd_dataset = load_msgd_dataset(msgd_dataset_path)

    # Initialize API clients
    openai_client = None
    gemini_llm = None

    if model.startswith("gemini-"):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini API not available. Install with: pip install google-genai")

        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        if not os.getenv("GCP_PROJECT_NAME"):
            raise ValueError("GCP_PROJECT_NAME environment variable not set!")

        gemini_llm = get_llm(model, series="gemini")
        print(f"Using Gemini API with model: {model}")
    else:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI API not available. Install with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")

        openai_client = OpenAI(api_key=api_key)
        print(f"Using OpenAI API with model: {model}")

    # Find all transcript files
    transcript_dir_path = Path(transcript_dir)
    transcript_files = sorted(transcript_dir_path.glob("*/transcript.json"))

    if max_dialogues:
        transcript_files = transcript_files[:max_dialogues]

    print(f"\nFound {len(transcript_files)} transcripts to evaluate")
    print(f"Output file: {output_file}\n")

    # Prepare output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_rounds = 0

    # Process each transcript
    for transcript_path in tqdm(transcript_files, desc="Evaluating dialogues"):
        dialogue_id = transcript_path.parent.name

        try:
            # Load transcript (with MSGD fallback if needed)
            transcript = load_transcript_with_fallback(transcript_path, msgd_dataset)

            # Extract rounds for per-round evaluation
            rounds = get_dialogue_rounds(transcript["dialogue_history"])

            if not rounds:
                print(f"  Warning: No rounds found in dialogue {dialogue_id}")
                continue

            # Evaluate each round
            round_evaluations = []
            for round_data in rounds:
                eval_result = evaluate_round(
                    user_turn=round_data["user_turn"],
                    agent_turn=round_data["agent_turn"],
                    history=round_data["history"],
                    model=model,
                    openai_client=openai_client,
                    gemini_llm=gemini_llm
                )

                round_evaluations.append({
                    "round_num": round_data["round_num"],
                    "user_turn": round_data["user_turn"][:100] + "..." if len(round_data["user_turn"]) > 100 else round_data["user_turn"],
                    "agent_turn": round_data["agent_turn"][:100] + "..." if len(round_data["agent_turn"]) > 100 else round_data["agent_turn"],
                    "evaluation": eval_result
                })

                total_rounds += 1

                # Rate limiting
                time.sleep(rate_limit_delay)

            # Calculate average scores for this dialogue
            avg_naturalness = sum(
                r["evaluation"]["naturalness"]["score"]
                for r in round_evaluations
                if "naturalness" in r["evaluation"]
            ) / max(len(round_evaluations), 1)

            avg_consistency = sum(
                r["evaluation"]["consistency"]["score"]
                for r in round_evaluations
                if "consistency" in r["evaluation"]
            ) / max(len(round_evaluations), 1)

            # Store result
            result = {
                "dialogue_id": dialogue_id,
                "agent_type": transcript.get("agent_type"),
                "total_rounds": len(rounds),
                "salesbot_context": transcript.get("salesbot_context"),
                "round_evaluations": round_evaluations,
                "average_scores": {
                    "naturalness": avg_naturalness,
                    "consistency": avg_consistency
                }
            }
            results.append(result)

            # Write progress incrementally
            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

        except Exception as e:
            print(f"\nError processing {dialogue_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "dialogue_id": dialogue_id,
                "error": str(e)
            })

            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

    # Print summary
    successful = sum(1 for r in results if "round_evaluations" in r)
    all_naturalness_scores = []
    all_consistency_scores = []

    for r in results:
        if "round_evaluations" in r:
            for round_eval in r["round_evaluations"]:
                if "naturalness" in round_eval["evaluation"]:
                    all_naturalness_scores.append(round_eval["evaluation"]["naturalness"]["score"])
                if "consistency" in round_eval["evaluation"]:
                    all_consistency_scores.append(round_eval["evaluation"]["consistency"]["score"])

    avg_naturalness = sum(all_naturalness_scores) / max(len(all_naturalness_scores), 1)
    avg_consistency = sum(all_consistency_scores) / max(len(all_consistency_scores), 1)

    print(f"\nWrote {total_rounds} round evaluations ({successful} dialogues) to {output_file}")
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total dialogues: {len(results)}")
    print(f"Successfully evaluated: {successful}")
    print(f"Total rounds evaluated: {total_rounds}")
    print(f"Average Naturalness: {avg_naturalness:.2f}")
    print(f"Average Consistency: {avg_consistency:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transcripts with GPT-4 or Gemini (per-round evaluation)"
    )

    parser.add_argument(
        "--transcript_dir",
        type=str,
        required=True,
        help="Directory containing transcript subdirectories"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--msgd_dataset",
        type=str,
        default=None,
        help="Path to MSGD_dataset_final.json (for audio transcripts with missing user text)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use: gpt-4o-mini, gpt-4, gemini-2.5-flash, etc. (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Max number of dialogues to evaluate (default: all)"
    )

    args = parser.parse_args()

    # Check API key based on model
    if args.model.startswith("gemini-"):
        if not os.getenv("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY environment variable not set!")
            print("Please run: export GEMINI_API_KEY='your-key-here'")
            return
        if not os.getenv("GCP_PROJECT_NAME"):
            print("ERROR: GCP_PROJECT_NAME environment variable not set!")
            print("Please run: export GCP_PROJECT_NAME='your-gcp-project'")
            return
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set!")
            print("Please run: export OPENAI_API_KEY='your-key-here'")
            return

    evaluate_transcript_directory(
        transcript_dir=args.transcript_dir,
        output_file=args.output_file,
        msgd_dataset_path=args.msgd_dataset,
        model=args.model,
        rate_limit_delay=args.rate_limit_delay,
        max_dialogues=args.max_dialogues
    )


if __name__ == "__main__":
    main()
