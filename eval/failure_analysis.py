"""
Classify dialogue failure patterns using LLM as judge

This script analyzes agent responses and classifies them into failure patterns:
1. Generic responses - Ignoring user's specific context
2. Topic switching - Abruptly changing subject
3. Repetition - Repeating same phrases without progression
4. Misunderstanding - Misinterpreting user's simple questions
5. Lack of memory - Forgetting previous turns in conversation

Usage:
  # With GPT-4o-mini
  python eval/failure_analysis.py \
    --transcript_dir dialogue_outputs/agent_omni_3b-ethan_fixed_eval \
    --output_file eval_results/failure_patterns_omni-3b.json \
    --model gpt-4o-mini

  # With Gemini 2.5 Flash
  python eval/failure_analysis.py \
    --transcript_dir dialogue_outputs/agent_mimo-audio_fixed_eval \
    --msgd_dataset data/MSGD_dataset_final.json \
    --output_file eval_results/failure_patterns_mimo.json \
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


# Context and instructions
CONTEXT = """
You are analyzing a conversation between a user and a salesbot agent. Your task is to identify failure patterns in the agent's responses.

A good agent response should:
- Address the user's specific question or comment
- Stay on topic unless the user changes it
- Progress the conversation forward
- Remember previous context
- Show understanding of user intent

When an agent fails, it typically exhibits one or more of these patterns:
1. **Generic responses** - Gives vague, template-like responses that ignore the user's specific context
2. **Topic switching** - Abruptly changes the subject without addressing what the user said
3. **Repetition** - Repeats the same phrases or questions without making progress
4. **Misunderstanding** - Misinterprets or fails to comprehend simple user questions
5. **Lack of memory** - Forgets information from previous turns in the conversation
"""

EVAL_SCHEMA = """
For each failure pattern, you must:
1. Determine if the pattern is present (true/false)
2. Provide a severity score (0-100):
   - 0: Pattern not present
   - 1-33: Mild issue (minor but noticeable)
   - 34-66: Moderate issue (clearly problematic)
   - 67-100: Severe issue (conversation-breaking)
3. Provide a brief reason explaining your assessment

Pattern Definitions:

**Generic Response**: The agent gives vague, non-specific responses that could apply to any situation,
ignoring the user's specific context, details, or questions. Examples:
- "How can I help you today?" when user already asked a specific question
- "That sounds interesting" without addressing specifics
- Template responses that don't acknowledge user's actual input

**Topic Switching**: The agent abruptly changes the subject or introduces new topics without properly
addressing the user's current input. Examples:
- User asks about movies, agent starts talking about restaurants
- User is in the middle of discussing preferences, agent jumps to unrelated service
- Agent pivots away from user's question without answering it

**Repetition**: The agent repeats the same phrases, questions, or responses that were already said
earlier, showing no progression. Examples:
- Asking "What can I help you with?" multiple times
- Repeating the same follow-up question after user already answered
- Giving identical responses to different user inputs

**Misunderstanding**: The agent misinterprets the user's clear and simple questions or statements,
showing failure to comprehend the basic meaning. Examples:
- User says "yes" and agent acts like they said "no"
- User asks for a specific item type, agent suggests wrong category
- Agent mistakes user's intent in obvious context

**Lack of Memory**: The agent forgets information that was mentioned in previous turns of the
conversation. Examples:
- User mentions their preference earlier, agent asks for it again
- Agent forgets which topic they were discussing
- Agent repeats suggestions that user already rejected

Output format (JSON):
{
    "generic_response": {
        "present": <true/false>,
        "severity": <0-100>,
        "reason": "<brief explanation>"
    },
    "topic_switching": {
        "present": <true/false>,
        "severity": <0-100>,
        "reason": "<brief explanation>"
    },
    "repetition": {
        "present": <true/false>,
        "severity": <0-100>,
        "reason": "<brief explanation>"
    },
    "misunderstanding": {
        "present": <true/false>,
        "severity": <0-100>,
        "reason": "<brief explanation>"
    },
    "lack_of_memory": {
        "present": <true/false>,
        "severity": <0-100>,
        "reason": "<brief explanation>"
    },
    "overall_assessment": {
        "has_failure": <true/false>,
        "primary_failure": "<pattern name or 'none'>",
        "explanation": "<brief overall assessment>"
    }
}
"""

TEMPLATE = """
{context}

Analyze the following dialogue turn and identify any failure patterns in the agent's response.

Dialogue History:
{history}

Current Turn:
User: {user_turn}
Agent: {agent_turn}

Instructions:
{eval_schema}

Output (JSON only, no other text):
"""


def load_msgd_dataset(dataset_path: str) -> Dict[str, Dict]:
    """Load MSGD dataset and organize by dialogue ID"""
    print(f"Loading MSGD dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    dialogues = {}
    for item in dataset:
        dialogue_id = item["id"].replace("merge_", "")
        dialogues[dialogue_id] = item

    print(f"✓ Loaded {len(dialogues)} dialogues from MSGD dataset")
    return dialogues


def get_user_text_from_msgd(msgd_data: Dict, turn_index: int) -> Optional[str]:
    """Extract user text from MSGD dataset for a specific turn"""
    dialog_turns = msgd_data.get("dialog", [])

    user_turn_count = 0
    target_user_turn = turn_index // 2

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
    """Load transcript and fill in missing user text from MSGD dataset"""
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    dialogue_id = transcript.get("dialogue_id")
    dialogue_history = transcript.get("dialogue_history", [])

    needs_fallback = any(
        turn.get("role") == "user" and
        (turn.get("content") == "[Audio input - transcription unavailable]" or
         not turn.get("content"))
        for turn in dialogue_history
    )

    if needs_fallback and msgd_dataset and dialogue_id in msgd_dataset:
        msgd_data = msgd_dataset[dialogue_id]

        for i, turn in enumerate(dialogue_history):
            if turn.get("role") == "user":
                if (turn.get("content") == "[Audio input - transcription unavailable]" or
                    not turn.get("content")):
                    user_text = get_user_text_from_msgd(msgd_data, i)
                    if user_text:
                        turn["content"] = user_text

    return transcript


def format_dialogue_history(dialogue_history: List[Dict[str, str]], up_to_index: int) -> str:
    """Format dialogue history up to a certain index"""
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
    """Extract dialogue rounds (user + agent pairs) for per-round evaluation"""
    rounds = []
    round_num = 0

    i = 0
    while i < len(dialogue_history) - 1:
        if dialogue_history[i].get("role") == "user":
            user_turn = dialogue_history[i].get("content", "")

            if i + 1 < len(dialogue_history) and dialogue_history[i + 1].get("role") == "agent":
                agent_turn = dialogue_history[i + 1].get("content", "")
                history = format_dialogue_history(dialogue_history, i)

                rounds.append({
                    "round_num": round_num,
                    "user_turn": user_turn,
                    "agent_turn": agent_turn,
                    "history": history,
                    "history_index": i
                })

                round_num += 1
                i += 2
            else:
                i += 1
        else:
            i += 1

    return rounds


def classify_patterns_with_openai(
    user_turn: str,
    agent_turn: str,
    history: str,
    client: OpenAI,
    model: str
) -> Dict:
    """Classify failure patterns in a single round using OpenAI API"""
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
            "generic_response": {"present": False, "severity": 0, "reason": "Parse error"},
            "topic_switching": {"present": False, "severity": 0, "reason": "Parse error"},
            "repetition": {"present": False, "severity": 0, "reason": "Parse error"},
            "misunderstanding": {"present": False, "severity": 0, "reason": "Parse error"},
            "lack_of_memory": {"present": False, "severity": 0, "reason": "Parse error"},
            "overall_assessment": {"has_failure": False, "primary_failure": "parse_error", "explanation": str(e)},
            "parse_error": str(e)
        }
    except Exception as e:
        print(f"  Error during OpenAI evaluation: {e}")
        return {
            "generic_response": {"present": False, "severity": 0, "reason": "API error"},
            "topic_switching": {"present": False, "severity": 0, "reason": "API error"},
            "repetition": {"present": False, "severity": 0, "reason": "API error"},
            "misunderstanding": {"present": False, "severity": 0, "reason": "API error"},
            "lack_of_memory": {"present": False, "severity": 0, "reason": "API error"},
            "overall_assessment": {"has_failure": False, "primary_failure": "api_error", "explanation": str(e)},
            "error": str(e)
        }


def classify_patterns_with_gemini(
    user_turn: str,
    agent_turn: str,
    history: str,
    model_name: str,
    llm_instance=None
) -> Dict:
    """Classify failure patterns in a single round using Gemini API"""
    prompt = TEMPLATE.format(
        context=CONTEXT,
        eval_schema=EVAL_SCHEMA,
        history=history,
        user_turn=user_turn,
        agent_turn=agent_turn
    )

    try:
        if llm_instance is None:
            llm_instance = get_llm(model_name, series="gemini")

        pattern_schema = gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "present": gtypes.Schema(type=gtypes.Type.BOOLEAN),
                "severity": gtypes.Schema(type=gtypes.Type.INTEGER),
                "reason": gtypes.Schema(type=gtypes.Type.STRING),
            },
            required=["present", "severity", "reason"],
        )

        overall_schema = gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "has_failure": gtypes.Schema(type=gtypes.Type.BOOLEAN),
                "primary_failure": gtypes.Schema(type=gtypes.Type.STRING),
                "explanation": gtypes.Schema(type=gtypes.Type.STRING),
            },
            required=["has_failure", "primary_failure", "explanation"],
        )

        response_schema = gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "generic_response": pattern_schema,
                "topic_switching": pattern_schema,
                "repetition": pattern_schema,
                "misunderstanding": pattern_schema,
                "lack_of_memory": pattern_schema,
                "overall_assessment": overall_schema,
            },
            required=["generic_response", "topic_switching", "repetition",
                     "misunderstanding", "lack_of_memory", "overall_assessment"],
        )

        conversations = [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "value": prompt}
                ]
            }
        ]

        response_text = llm_instance(
            conversations,
            temperature=0.0,
            max_tokens=2048,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        if isinstance(response_text, tuple):
            response_text = response_text[0]

        result_text = str(response_text).strip()

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
            "generic_response": {"present": False, "severity": 0, "reason": "Parse error"},
            "topic_switching": {"present": False, "severity": 0, "reason": "Parse error"},
            "repetition": {"present": False, "severity": 0, "reason": "Parse error"},
            "misunderstanding": {"present": False, "severity": 0, "reason": "Parse error"},
            "lack_of_memory": {"present": False, "severity": 0, "reason": "Parse error"},
            "overall_assessment": {"has_failure": False, "primary_failure": "parse_error", "explanation": str(e)},
            "parse_error": str(e)
        }
    except Exception as e:
        print(f"  Error during Gemini evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "generic_response": {"present": False, "severity": 0, "reason": "API error"},
            "topic_switching": {"present": False, "severity": 0, "reason": "API error"},
            "repetition": {"present": False, "severity": 0, "reason": "API error"},
            "misunderstanding": {"present": False, "severity": 0, "reason": "API error"},
            "lack_of_memory": {"present": False, "severity": 0, "reason": "API error"},
            "overall_assessment": {"has_failure": False, "primary_failure": "api_error", "explanation": str(e)},
            "error": str(e)
        }


def classify_patterns(
    user_turn: str,
    agent_turn: str,
    history: str,
    model: str,
    openai_client: Optional[OpenAI] = None,
    gemini_llm=None
) -> Dict:
    """Classify failure patterns using the appropriate API"""
    if model.startswith("gemini-") or model.startswith("models/gemini-"):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini API not available. Install with: pip install google-genai")
        return classify_patterns_with_gemini(user_turn, agent_turn, history, model, gemini_llm)
    else:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI API not available. Install with: pip install openai")
        if openai_client is None:
            raise ValueError("OpenAI client required for OpenAI models")
        return classify_patterns_with_openai(user_turn, agent_turn, history, openai_client, model)


def classify_transcript_directory(
    transcript_dir: str,
    output_file: str,
    msgd_dataset_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    rate_limit_delay: float = 1.0,
    max_dialogues: Optional[int] = None
):
    """Classify failure patterns for all transcripts in a directory"""
    msgd_dataset = None
    if msgd_dataset_path:
        msgd_dataset = load_msgd_dataset(msgd_dataset_path)

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

    transcript_dir_path = Path(transcript_dir)
    transcript_files = sorted(transcript_dir_path.glob("*/transcript.json"))

    if max_dialogues:
        transcript_files = transcript_files[:max_dialogues]

    print(f"\nFound {len(transcript_files)} transcripts to classify")
    print(f"Output file: {output_file}\n")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_rounds = 0
    pattern_counts = {
        "generic_response": 0,
        "topic_switching": 0,
        "repetition": 0,
        "misunderstanding": 0,
        "lack_of_memory": 0
    }

    for transcript_path in tqdm(transcript_files, desc="Classifying patterns"):
        dialogue_id = transcript_path.parent.name

        try:
            transcript = load_transcript_with_fallback(transcript_path, msgd_dataset)
            rounds = get_dialogue_rounds(transcript["dialogue_history"])

            if not rounds:
                print(f"  Warning: No rounds found in dialogue {dialogue_id}")
                continue

            round_classifications = []
            for round_data in rounds:
                classification = classify_patterns(
                    user_turn=round_data["user_turn"],
                    agent_turn=round_data["agent_turn"],
                    history=round_data["history"],
                    model=model,
                    openai_client=openai_client,
                    gemini_llm=gemini_llm
                )

                for pattern in pattern_counts.keys():
                    if classification.get(pattern, {}).get("present", False):
                        pattern_counts[pattern] += 1

                round_classifications.append({
                    "round_num": round_data["round_num"],
                    "user_turn": round_data["user_turn"],
                    "agent_turn": round_data["agent_turn"],
                    "classification": classification
                })

                total_rounds += 1
                time.sleep(rate_limit_delay)

            result = {
                "dialogue_id": dialogue_id,
                "agent_type": transcript.get("agent_type"),
                "total_rounds": len(rounds),
                "salesbot_context": transcript.get("salesbot_context"),
                "round_classifications": round_classifications
            }
            results.append(result)

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

    successful = sum(1 for r in results if "round_classifications" in r)

    print(f"\nWrote {total_rounds} round classifications ({successful} dialogues) to {output_file}")
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total dialogues: {len(results)}")
    print(f"Successfully classified: {successful}")
    print(f"Total rounds classified: {total_rounds}")
    print(f"\nPattern Occurrences:")
    for pattern, count in pattern_counts.items():
        pct = (count / total_rounds * 100) if total_rounds > 0 else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Classify dialogue failure patterns using LLM as judge"
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
        help="Path to MSGD_dataset_final.json (for audio transcripts)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use: gpt-4o-mini, gpt-4, gemini-2.5-flash, etc."
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Max number of dialogues to classify (default: all)"
    )

    args = parser.parse_args()

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

    classify_transcript_directory(
        transcript_dir=args.transcript_dir,
        output_file=args.output_file,
        msgd_dataset_path=args.msgd_dataset,
        model=args.model,
        rate_limit_delay=args.rate_limit_delay,
        max_dialogues=args.max_dialogues
    )


if __name__ == "__main__":
    main()
