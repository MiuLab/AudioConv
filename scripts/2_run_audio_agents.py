"""
Step 2: Run Audio Agents (Fixed-Input Evaluation)

Evaluates agent systems using pre-recorded user audio from data/user_audio/.
Instead of free-form dialogue rollout, it:
1. Uses fixed user audio inputs (merge_XXXX_turnYY.wav)
2. Generates only agent responses one round at a time
3. Saves output as: dialogue_outputs/<config>/XXXX/agent_audio_round_YYY.wav

This enables controlled comparison across different agent systems with identical user inputs.

Usage:
  # Modular agent (Whisper + GPT-4o-mini + Sesame CSM)
  python scripts/2_run_audio_agents.py \
    --agent_type sesame \
    --openai_model gpt-4o-mini \
    --user_audio_dir data/user_audio \
    --output_dir dialogue_outputs

  # Qwen2.5-Omni-7B (E2E)
  python scripts/2_run_audio_agents.py \
    --agent_type qwen_omni_7b \
    --speaker Chelsie \
    --user_audio_dir data/user_audio \
    --output_dir dialogue_outputs

  # MiMo-Audio-7B (E2E)
  python scripts/2_run_audio_agents.py \
    --agent_type mimo \
    --user_audio_dir data/user_audio \
    --output_dir dialogue_outputs
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re

from audioconv.agents.modular_agent import AgentSystem
from audioconv.agents.qwen_omni_agent import AgentSystemOmni
from audioconv.agents.mimo_agent import AgentSystemMiMo


def get_all_dialogue_ids(user_audio_dir: str = "data/user_audio") -> List[str]:
    """Get all unique dialogue IDs from user audio directory."""
    audio_dir = Path(user_audio_dir)
    dialogue_ids = set()

    for audio_file in audio_dir.glob("merge_*_turn*.wav"):
        match = re.search(r'merge_(\d+)_turn', audio_file.name)
        if match:
            dialogue_ids.add(match.group(1))

    return sorted(dialogue_ids)


class FixedInputEvaluator:
    """Evaluator for agent systems using fixed user audio inputs."""

    def __init__(
        self,
        agent_type: str,
        agent_config: Optional[Dict] = None,
        user_audio_dir: str = "data/user_audio",
        output_base_dir: str = "dialogue_outputs",
        use_audio_history: Optional[bool] = None,
        whisper_device: Optional[str] = None
    ):
        """
        Initialize fixed-input evaluator.

        Args:
            agent_type: One of "sesame", "qwen_omni_3b", "qwen_omni_7b", "mimo"
            agent_config: Additional config (speaker_id, model params, etc.)
            user_audio_dir: Directory containing user audio files
            output_base_dir: Base directory for outputs
            use_audio_history: Whether to pass audio files in dialogue history
                None = Auto-detect (True for audio-native models, False for Sesame)
            whisper_device: Device for Whisper transcription (e.g., "cuda:1", "cpu")
        """
        self.agent_type = agent_type
        self.agent_config = agent_config or {}
        self.user_audio_dir = Path(user_audio_dir)
        self.output_base_dir = Path(output_base_dir)
        self.whisper_device = whisper_device

        # Auto-detect audio history mode
        if use_audio_history is None:
            self.use_audio_history = agent_type in ["qwen_omni_3b", "qwen_omni_7b", "mimo"]
        else:
            self.use_audio_history = use_audio_history

        print(f"Audio history mode: {'ENABLED' if self.use_audio_history else 'DISABLED (text-only)'}")

        self.agent = self._initialize_agent()
        self.output_dir = self._create_output_directory()

    def _initialize_agent(self):
        """Initialize the appropriate agent system."""
        print(f"Initializing agent: {self.agent_type}")

        if self.agent_type == "sesame":
            whisper_model = self.agent_config.get("whisper_model", "base")
            openai_model = self.agent_config.get("openai_model", "gpt-4o-mini")
            speaker_id = self.agent_config.get("speaker_id", "1")

            if self.whisper_device:
                print(f"Sesame: Using separate Whisper on {self.whisper_device}")
                agent = AgentSystem(
                    whisper_model=whisper_model,
                    openai_model=openai_model,
                    device="cuda:0"
                )
                import whisper
                print(f"Loading Whisper ({whisper_model}) on {self.whisper_device} for STT...")
                agent.whisper = whisper.load_model(whisper_model, device=self.whisper_device)
            else:
                agent = AgentSystem(
                    whisper_model=whisper_model,
                    openai_model=openai_model
                )

            agent.load_voice_profile(speaker_id=speaker_id)
            return agent

        elif self.agent_type.startswith("qwen_omni"):
            model_size = "3B" if "3b" in self.agent_type else "7B"
            model_id = f"Qwen/Qwen2.5-Omni-{model_size}"
            speaker = self.agent_config.get("speaker", "Ethan")

            return AgentSystemOmni(
                model_id=model_id,
                speaker=speaker,
                quantized=True
            )

        elif self.agent_type == "mimo":
            return AgentSystemMiMo(
                model_id="XiaomiMiMo/MiMo-Audio-7B-Instruct"
            )

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def _create_output_directory(self) -> Path:
        """Create output directory based on agent configuration."""
        if self.agent_type == "sesame":
            speaker = self.agent_config.get("speaker_id", "1")
            model = self.agent_config.get("openai_model", "gpt-4o-mini")
            dir_name = f"agent_sesame_{model}_speaker{speaker}_fixed_eval"

        elif self.agent_type.startswith("qwen_omni"):
            size = "3b" if "3b" in self.agent_type else "7b"
            speaker = self.agent_config.get("speaker", "ethan").lower()
            dir_name = f"agent_omni_{size}-{speaker}_fixed_eval"

        elif self.agent_type == "mimo":
            dir_name = "agent_mimo-audio_fixed_eval"

        output_dir = self.output_base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")
        return output_dir

    def _get_user_transcription(self, user_audio_path: str) -> str:
        """Get user audio transcription."""
        if hasattr(self.agent, 'speech_to_text'):
            return self.agent.speech_to_text(user_audio_path)
        else:
            try:
                import whisper
                import torch

                if not hasattr(self, '_whisper_model'):
                    if self.whisper_device:
                        device = self.whisper_device
                    else:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Loading Whisper (base) for transcription logging on {device}...")
                    self._whisper_model = whisper.load_model("base", device=device)
                    self._whisper_device = device

                result = self._whisper_model.transcribe(user_audio_path)
                return result["text"].strip()
            except Exception as e:
                print(f"Warning: Could not transcribe for logging: {e}")
                return "[Audio input - transcription unavailable]"

    def get_user_audio_files(self, dialogue_id: str) -> List[Path]:
        """Get all user audio files for a specific dialogue ID."""
        pattern = f"merge_{dialogue_id}_turn*.wav"
        files = sorted(self.user_audio_dir.glob(pattern))

        user_files = []
        for f in files:
            turn_match = re.search(r'turn(\d+)', f.name)
            if turn_match:
                turn_num = int(turn_match.group(1))
                # User turns are even numbered (00, 02, 04, ...)
                if turn_num % 2 == 0:
                    user_files.append(f)

        return user_files

    def process_single_round(
        self,
        user_audio_path: Path,
        round_number: int,
        dialogue_history: List[Dict],
        salesbot_context: Optional[Dict] = None,
        dialogue_output_dir: Optional[Path] = None
    ) -> Dict:
        """Process a single round: user audio → agent response."""
        print(f"\n{'='*60}")
        print(f"Processing Round {round_number}")
        print(f"User audio: {user_audio_path.name}")
        print(f"{'='*60}")

        output_dir = dialogue_output_dir if dialogue_output_dir else self.output_dir
        output_audio_path = output_dir / f"agent_audio_round_{round_number:03d}.wav"

        process_kwargs = {
            "user_audio_path": str(user_audio_path),
            "output_audio_path": str(output_audio_path),
            "dialogue_history": dialogue_history,
            "salesbot_context": salesbot_context
        }

        if self.agent_type in ["qwen_omni_3b", "qwen_omni_7b", "mimo"]:
            process_kwargs["use_audio_history"] = self.use_audio_history

        result = self.agent.process_turn(**process_kwargs)

        print(f"Agent audio saved: {output_audio_path.name}")
        return result

    def process_dialogue(
        self,
        dialogue_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None,
        salesbot_context: Optional[Dict] = None,
        save_transcript: bool = True
    ) -> List[Dict]:
        """Process multiple rounds of a dialogue."""
        print(f"\n{'#'*60}")
        print(f"Processing Dialogue: {dialogue_id}")
        print(f"{'#'*60}")

        dialogue_output_dir = self.output_dir / dialogue_id
        dialogue_output_dir.mkdir(parents=True, exist_ok=True)

        user_files = self.get_user_audio_files(dialogue_id)

        if not user_files:
            print(f"No user audio files found for dialogue {dialogue_id}")
            return []

        if end_round is not None:
            user_files = user_files[start_round:end_round+1]
        else:
            user_files = user_files[start_round:]

        # Check for existing progress
        completed_rounds = self._get_completed_rounds(dialogue_id)

        if completed_rounds > 0:
            print(f"Found existing progress with {completed_rounds} completed rounds")
            dialogue_history = self._load_existing_dialogue_history(dialogue_id)

            last_audio = dialogue_output_dir / f"agent_audio_round_{completed_rounds-1:03d}.wav"
            if not last_audio.exists():
                print(f"Warning: Audio file missing for round {completed_rounds-1}, reprocessing from round 0")
                completed_rounds = 0
                dialogue_history = []
            else:
                if completed_rounds < len(user_files):
                    user_files = user_files[completed_rounds:]
                    start_round = completed_rounds
                    print(f"Resuming from round {start_round} ({len(user_files)} rounds remaining)")
                else:
                    print(f"Dialogue already complete with {completed_rounds} rounds")
                    return []

        if completed_rounds == 0:
            if dialogue_output_dir.exists():
                for f in dialogue_output_dir.glob("agent_audio_round_*.wav"):
                    print(f"Removing incomplete audio: {f.name}")
                    f.unlink()
                transcript_file = dialogue_output_dir / "transcript.json"
                if transcript_file.exists():
                    transcript_file.unlink()
            dialogue_history = []

        print(f"Processing {len(user_files)} rounds (starting from round {start_round})")

        agent_responses = []

        for idx, user_audio_path in enumerate(user_files):
            round_num = start_round + idx

            print(f"\nTranscribing user audio...")
            user_text = self._get_user_transcription(str(user_audio_path))
            print(f"User said: {user_text}")

            dialogue_history.append({
                "role": "user",
                "content": user_text,
                "audio_path": str(user_audio_path)
            })

            result = self.process_single_round(
                user_audio_path=user_audio_path,
                round_number=idx,
                dialogue_history=dialogue_history[:-1],
                salesbot_context=salesbot_context,
                dialogue_output_dir=dialogue_output_dir
            )

            dialogue_history.append({
                "role": "agent",
                "content": result["text"],
                "audio_path": result["audio_path"]
            })

            agent_responses.append(result)

        if save_transcript:
            transcript_path = dialogue_output_dir / "transcript.json"
            total_rounds = len([t for t in dialogue_history if t.get("role") == "agent"])

            transcript = {
                "dialogue_id": dialogue_id,
                "agent_type": self.agent_type,
                "agent_config": self.agent_config,
                "total_rounds": total_rounds,
                "salesbot_context": salesbot_context,
                "dialogue_history": dialogue_history
            }

            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)

            print(f"\nTranscript saved: {transcript_path} (total rounds: {total_rounds})")

        return agent_responses

    def _get_completed_rounds(self, dialogue_id: str) -> int:
        """Get the number of completed rounds for a dialogue."""
        dialogue_output_dir = self.output_dir / dialogue_id
        transcript_path = dialogue_output_dir / "transcript.json"

        if not transcript_path.exists():
            return 0

        try:
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
            completed = transcript.get("total_rounds", 0)

            actual_audio_files = 0
            for i in range(completed):
                audio_path = dialogue_output_dir / f"agent_audio_round_{i:03d}.wav"
                if audio_path.exists():
                    actual_audio_files += 1
                else:
                    break

            return min(completed, actual_audio_files)
        except Exception as e:
            print(f"Warning: Error reading transcript for {dialogue_id}: {e}")
            return 0

    def _load_existing_dialogue_history(self, dialogue_id: str) -> List[Dict]:
        """Load existing dialogue history from transcript."""
        dialogue_output_dir = self.output_dir / dialogue_id
        transcript_path = dialogue_output_dir / "transcript.json"

        if not transcript_path.exists():
            return []

        try:
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
            return transcript.get("dialogue_history", [])
        except Exception as e:
            print(f"Warning: Could not load existing transcript: {e}")
            return []

    def process_batch(
        self,
        dialogue_ids: List[str],
        rounds_per_dialogue: int = None,
        salesbot_contexts: Optional[Dict[str, Dict]] = None,
        skip_existing: bool = True
    ):
        """Process multiple dialogues in batch."""
        print(f"\n{'#'*60}")
        print(f"BATCH PROCESSING: {len(dialogue_ids)} dialogues")
        if skip_existing:
            print("Skip existing: ENABLED (will resume from incomplete dialogues)")
        else:
            print("Skip existing: DISABLED (will reprocess all dialogues)")
        print(f"{'#'*60}\n")

        if skip_existing:
            dialogue_ids_filtered = []
            fully_completed = 0
            partially_completed = 0

            for did in dialogue_ids:
                completed_rounds = self._get_completed_rounds(did)
                expected_rounds = rounds_per_dialogue if rounds_per_dialogue else 999

                if completed_rounds == 0:
                    dialogue_ids_filtered.append(did)
                elif completed_rounds < expected_rounds:
                    dialogue_ids_filtered.append(did)
                    partially_completed += 1
                else:
                    fully_completed += 1

            dialogue_ids = dialogue_ids_filtered

            if fully_completed > 0:
                print(f"Skipping {fully_completed} fully-completed dialogues")
            if partially_completed > 0:
                print(f"Resuming {partially_completed} partially-completed dialogues")
            print(f"Total to process: {len(dialogue_ids)} dialogues\n")

        for dialogue_id in dialogue_ids:
            salesbot_context = None
            if salesbot_contexts and dialogue_id in salesbot_contexts:
                salesbot_context = salesbot_contexts[dialogue_id]

            try:
                end_round = rounds_per_dialogue - 1 if rounds_per_dialogue else None
                self.process_dialogue(
                    dialogue_id=dialogue_id,
                    start_round=0,
                    end_round=end_round,
                    salesbot_context=salesbot_context,
                    save_transcript=True
                )
            except Exception as e:
                print(f"Error processing dialogue {dialogue_id}: {e}")
                continue

        print(f"\n{'#'*60}")
        print(f"BATCH COMPLETE")
        print(f"{'#'*60}")


def main():
    parser = argparse.ArgumentParser(description="Fixed-input audio agent evaluation")

    parser.add_argument("--agent_type", type=str, required=True,
                       choices=["sesame", "qwen_omni_3b", "qwen_omni_7b", "mimo"],
                       help="Agent system type")
    parser.add_argument("--speaker_id", type=str, default="1",
                       help="Speaker ID for Sesame agent")
    parser.add_argument("--speaker", type=str, default="Ethan",
                       help="Speaker name for Qwen Omni (Ethan/Chelsie)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model for LLM reasoning (Sesame agent)")
    parser.add_argument("--whisper_model", type=str, default="base",
                       help="Whisper model size for STT")
    parser.add_argument("--whisper_device", type=str, default=None,
                       help="Device for Whisper transcription (e.g., 'cuda:1', 'cpu')")

    parser.add_argument("--dialogue_ids", type=str, nargs="*", default=None,
                       help="Dialogue IDs to process. Omit to process all available dialogues.")
    parser.add_argument("--first_n", type=int, default=None,
                       help="Only process first N dialogues")
    parser.add_argument("--rounds", type=int, default=None,
                       help="Max rounds per dialogue (None for all)")
    parser.add_argument("--user_audio_dir", type=str,
                       default="data/user_audio",
                       help="Directory containing user audio files")
    parser.add_argument("--output_dir", type=str,
                       default="dialogue_outputs",
                       help="Base output directory")

    parser.add_argument("--intent", type=str, default=None,
                       help="SalesBot intent (e.g., FindMovie, FindRestaurants)")
    parser.add_argument("--domain", type=str, default=None,
                       help="SalesBot domain")

    parser.add_argument("--use_audio_history", action="store_true",
                       help="Pass audio files in dialogue history (auto-enabled for audio-native models)")
    parser.add_argument("--no_audio_history", action="store_true",
                       help="Force text-only history even for audio-native models")

    parser.add_argument("--no_skip_existing", action="store_true",
                       help="Reprocess all dialogues even if already completed")

    args = parser.parse_args()

    agent_config = {
        "speaker_id": args.speaker_id,
        "speaker": args.speaker,
        "openai_model": args.openai_model,
        "whisper_model": args.whisper_model
    }

    salesbot_context = None
    if args.intent or args.domain:
        salesbot_context = {}
        if args.intent:
            salesbot_context["intent"] = args.intent
        if args.domain:
            salesbot_context["domain"] = args.domain

    if args.dialogue_ids is None or len(args.dialogue_ids) == 0 or \
       (len(args.dialogue_ids) == 1 and args.dialogue_ids[0].lower() == "all"):
        print("Scanning for all available dialogue IDs...")
        dialogue_ids = get_all_dialogue_ids(args.user_audio_dir)
        print(f"Found {len(dialogue_ids)} unique dialogue IDs")
        if args.first_n:
            dialogue_ids = dialogue_ids[:args.first_n]
            print(f"Processing first {len(dialogue_ids)} dialogues")
    else:
        dialogue_ids = args.dialogue_ids
        print(f"Processing {len(dialogue_ids)} specified dialogues")

    use_audio_history = None
    if args.use_audio_history:
        use_audio_history = True
    elif args.no_audio_history:
        use_audio_history = False

    evaluator = FixedInputEvaluator(
        agent_type=args.agent_type,
        agent_config=agent_config,
        user_audio_dir=args.user_audio_dir,
        output_base_dir=args.output_dir,
        use_audio_history=use_audio_history,
        whisper_device=args.whisper_device
    )

    evaluator.process_batch(
        dialogue_ids=dialogue_ids,
        rounds_per_dialogue=args.rounds,
        salesbot_contexts={did: salesbot_context for did in dialogue_ids},
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    main()
