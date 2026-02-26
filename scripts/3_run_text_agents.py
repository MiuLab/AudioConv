"""
Step 3: Run Text-Only Agent Evaluation

Evaluates agent systems using user text inputs from MSGD_dataset_final.json.
All models run in text-only mode (no audio). This provides the text-only
baseline for comparison with audio agents.

Supported agent types:
- gpt4o_mini: GPT-4o-mini (pure text baseline)
- qwen_omni_3b: Qwen2.5-Omni-3B (text-only mode, no audio)
- qwen_omni_7b: Qwen2.5-Omni-7B (text-only mode, no audio)
- mimo: MiMo-Audio-7B (text-only mode, no audio)

Usage:
  # GPT-4o-mini baseline
  python scripts/3_run_text_agents.py \
    --agent_type gpt4o_mini \
    --dataset data/MSGD_dataset_final.json \
    --output_dir dialogue_outputs_text_only

  # Qwen2.5-Omni-3B text-only
  python scripts/3_run_text_agents.py \
    --agent_type qwen_omni_3b \
    --dataset data/MSGD_dataset_final.json \
    --output_dir dialogue_outputs_text_only
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI


def load_msgd_dataset(dataset_path: str = "data/MSGD_dataset_final.json") -> Dict[str, Dict]:
    """Load MSGD dataset and organize by dialogue ID."""
    print(f"Loading MSGD dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    dialogues = {}
    for item in dataset:
        dialogue_id = item["id"].replace("merge_", "")
        dialogues[dialogue_id] = item

    print(f"✓ Loaded {len(dialogues)} dialogues from MSGD dataset")
    return dialogues


class TextOnlyEvaluator:
    """Evaluator for agent systems using text-only inputs."""

    def __init__(
        self,
        agent_type: str,
        agent_config: Optional[Dict] = None,
        msgd_dataset: Optional[Dict[str, Dict]] = None,
        output_base_dir: str = "dialogue_outputs_text_only"
    ):
        self.agent_type = agent_type
        self.agent_config = agent_config or {}
        self.msgd_dataset = msgd_dataset or {}
        self.output_base_dir = Path(output_base_dir)

        print(f"Text-only evaluation mode: {agent_type}")

        self.agent = self._initialize_agent()
        self.output_dir = self._create_output_directory()

    def _initialize_agent(self):
        """Initialize the appropriate agent system (text-only mode)."""
        print(f"Initializing text-only agent: {self.agent_type}")

        if self.agent_type == "gpt4o_mini":
            return TextOnlyGPTAgent(
                model=self.agent_config.get("model", "gpt-4o-mini")
            )

        elif self.agent_type.startswith("qwen_omni"):
            model_size = "3B" if "3b" in self.agent_type else "7B"
            model_id = f"Qwen/Qwen2.5-Omni-{model_size}"
            return TextOnlyQwenOmniAgent(model_id=model_id, quantized=True)

        elif self.agent_type == "mimo":
            return TextOnlyMiMoAgent(model_id="XiaomiMiMo/MiMo-Audio-7B-Instruct")

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def _create_output_directory(self) -> Path:
        """Create output directory based on agent configuration."""
        if self.agent_type == "gpt4o_mini":
            model = self.agent_config.get("model", "gpt-4o-mini")
            dir_name = f"agent_{model}_text_only_fixed_eval"

        elif self.agent_type.startswith("qwen_omni"):
            size = "3b" if "3b" in self.agent_type else "7b"
            dir_name = f"agent_omni_{size}_text_only_fixed_eval"

        elif self.agent_type == "mimo":
            dir_name = "agent_mimo_text_only_fixed_eval"

        output_dir = self.output_base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")
        return output_dir

    def get_user_text_inputs(self, dialogue_id: str) -> List[Dict]:
        """Get all user text inputs for a specific dialogue ID from MSGD dataset."""
        if dialogue_id not in self.msgd_dataset:
            print(f"Warning: Dialogue {dialogue_id} not found in MSGD dataset")
            return []

        dialogue_data = self.msgd_dataset[dialogue_id]
        dialog_turns = dialogue_data.get("dialog", [])

        user_inputs = []
        for i, turn in enumerate(dialog_turns):
            if turn.startswith("User:"):
                user_text = turn.replace("User:", "").strip()
                user_inputs.append({
                    'text': user_text,
                    'turn_number': i,
                    'original_turn': turn
                })

        return user_inputs

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

        if salesbot_context is None and dialogue_id in self.msgd_dataset:
            dialogue_data = self.msgd_dataset[dialogue_id]
            salesbot_context = {
                "intent": dialogue_data.get("intent", {}).get("type"),
                "intent_description": dialogue_data.get("intent", {}).get("description"),
                "transition_position": dialogue_data.get("transition_sentence", {}).get("position")
            }
            print(f"Loaded SalesBot context: {salesbot_context}")

        dialogue_output_dir = self.output_dir / dialogue_id
        dialogue_output_dir.mkdir(parents=True, exist_ok=True)

        user_inputs = self.get_user_text_inputs(dialogue_id)

        if not user_inputs:
            print(f"No user text inputs found for dialogue {dialogue_id}")
            return []

        if end_round is not None:
            user_inputs = user_inputs[start_round:end_round+1]
        else:
            user_inputs = user_inputs[start_round:]

        print(f"Processing {len(user_inputs)} rounds")

        dialogue_history = []
        agent_responses = []

        for idx, user_input in enumerate(user_inputs):
            user_text = user_input['text']
            print(f"\nRound {idx}: {user_text[:80]}...")

            dialogue_history.append({
                "role": "user",
                "content": user_text
            })

            result = self.agent.generate_response(
                user_text=user_text,
                dialogue_history=dialogue_history[:-1],
                salesbot_context=salesbot_context
            )

            print(f"Agent: {result['text'][:80]}...")

            dialogue_history.append({
                "role": "agent",
                "content": result["text"]
            })

            agent_responses.append(result)

        if save_transcript:
            transcript_path = dialogue_output_dir / "transcript.json"

            transcript = {
                "dialogue_id": dialogue_id,
                "agent_type": self.agent_type,
                "agent_config": self.agent_config,
                "total_rounds": len(agent_responses),
                "salesbot_context": salesbot_context,
                "dialogue_history": dialogue_history
            }

            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)

            print(f"\nTranscript saved: {transcript_path}")

        return agent_responses

    def process_batch(
        self,
        dialogue_ids: List[str],
        rounds_per_dialogue: int = None,
        salesbot_contexts: Optional[Dict[str, Dict]] = None
    ):
        """Process multiple dialogues in batch."""
        print(f"\n{'#'*60}")
        print(f"BATCH PROCESSING: {len(dialogue_ids)} dialogues")
        print(f"{'#'*60}\n")

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
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'#'*60}")
        print(f"BATCH COMPLETE")
        print(f"{'#'*60}")


class TextOnlyGPTAgent:
    """Pure text GPT agent (baseline)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = """You are a helpful, friendly sales assistant.
Have natural chitchat, understand the user's needs, and offer helpful, concrete suggestions.
Be conversational and empathetic. If possible, suggest attractions, restaurants, movies, hotels, or events based on context."""

        print(f"✓ Initialized GPT-{model} text-only agent")

    def generate_response(
        self,
        user_text: str,
        dialogue_history: List[Dict],
        salesbot_context: Optional[Dict] = None
    ) -> Dict:
        """Generate text response using GPT."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for turn in dialogue_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        agent_text = response.choices[0].message.content.strip()
        return {"text": agent_text, "role": "agent"}


class TextOnlyQwenOmniAgent:
    """Qwen Omni agent in text-only mode (no audio)."""

    def __init__(self, model_id: str, quantized: bool = True):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quantization_config = None
        if quantized:
            print("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        print(f"Loading {model_id} in text-only mode...")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            device_map=self.device,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model.disable_talker()

        self.system_prompt = """You are a helpful, friendly sales assistant.
Have natural chitchat, understand the user's needs, and offer helpful, concrete suggestions.
Be conversational and empathetic. If possible, suggest attractions, restaurants, movies, hotels, or events based on context."""

        print(f"✓ Initialized Qwen Omni text-only agent")

    def generate_response(
        self,
        user_text: str,
        dialogue_history: List[Dict],
        salesbot_context: Optional[Dict] = None
    ) -> Dict:
        """Generate text response using Qwen Omni (text-only)."""
        import torch

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            }
        ]

        for turn in dialogue_history[-6:]:
            role = "assistant" if turn["role"] == "agent" else "user"
            conversation.append({
                "role": role,
                "content": [{"type": "text", "text": turn["content"]}]
            })

        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}]
        })

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        agent_text = self.processor.decode(response_ids, skip_special_tokens=True).strip()

        return {"text": agent_text, "role": "agent"}


class TextOnlyMiMoAgent:
    """MiMo agent in text-only mode (no audio)."""

    def __init__(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_id} in text-only mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        self.system_prompt = """You are a helpful, friendly sales assistant.
Have natural chitchat, understand the user's needs, and offer helpful, concrete suggestions.
Be conversational and empathetic. If possible, suggest attractions, restaurants, movies, hotels, or events based on context."""

        print(f"✓ Initialized MiMo text-only agent")

    def generate_response(
        self,
        user_text: str,
        dialogue_history: List[Dict],
        salesbot_context: Optional[Dict] = None
    ) -> Dict:
        """Generate text response using MiMo (text-only)."""
        import re
        import torch

        conversation = [{"role": "system", "content": self.system_prompt}]

        for turn in dialogue_history[-6:]:
            role = "assistant" if turn["role"] == "agent" else "user"
            conversation.append({"role": role, "content": turn["content"]})

        conversation.append({"role": "user", "content": user_text})

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        agent_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if "</think>" in agent_text:
            agent_text = agent_text.split("</think>", 1)[-1].strip()
        elif "<think>" in agent_text:
            agent_text = re.sub(r"<think>.*", "", agent_text, flags=re.DOTALL).strip()

        agent_text = agent_text.replace("<|eot|>", "").strip()

        return {"text": agent_text, "role": "agent"}


def main():
    parser = argparse.ArgumentParser(description="Fixed-input text-only agent evaluation")

    parser.add_argument("--agent_type", type=str, required=True,
                       choices=["gpt4o_mini", "qwen_omni_3b", "qwen_omni_7b", "mimo"],
                       help="Agent system type (text-only)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model name for GPT agents")

    parser.add_argument("--dialogue_ids", type=str, nargs="*", default=None,
                       help="Dialogue IDs to process. Omit to process all.")
    parser.add_argument("--first_n", type=int, default=None,
                       help="Only process first N dialogues")
    parser.add_argument("--rounds", type=int, default=None,
                       help="Max rounds per dialogue (None for all)")
    parser.add_argument("--dataset", type=str,
                       default="data/MSGD_dataset_final.json",
                       help="Path to MSGD_dataset_final.json")
    parser.add_argument("--output_dir", type=str,
                       default="dialogue_outputs_text_only",
                       help="Base output directory")

    parser.add_argument("--intent", type=str, default=None,
                       help="SalesBot intent override (e.g., FindMovie)")
    parser.add_argument("--domain", type=str, default=None,
                       help="SalesBot domain override")

    args = parser.parse_args()

    msgd_dataset = load_msgd_dataset(args.dataset)

    agent_config = {"model": args.model}

    salesbot_context = None
    if args.intent or args.domain:
        salesbot_context = {}
        if args.intent:
            salesbot_context["intent"] = args.intent
        if args.domain:
            salesbot_context["domain"] = args.domain

    if args.dialogue_ids is None or len(args.dialogue_ids) == 0 or \
       (len(args.dialogue_ids) == 1 and args.dialogue_ids[0].lower() == "all"):
        print("Using all available dialogues from MSGD dataset...")
        dialogue_ids = sorted(msgd_dataset.keys())
        print(f"Found {len(dialogue_ids)} dialogues")
        if args.first_n:
            dialogue_ids = dialogue_ids[:args.first_n]
            print(f"Processing first {len(dialogue_ids)} dialogues")
    else:
        dialogue_ids = args.dialogue_ids
        print(f"Processing {len(dialogue_ids)} specified dialogues")

    evaluator = TextOnlyEvaluator(
        agent_type=args.agent_type,
        agent_config=agent_config,
        msgd_dataset=msgd_dataset,
        output_base_dir=args.output_dir
    )

    salesbot_contexts = None
    if salesbot_context:
        salesbot_contexts = {did: salesbot_context for did in dialogue_ids}

    evaluator.process_batch(
        dialogue_ids=dialogue_ids,
        rounds_per_dialogue=args.rounds,
        salesbot_contexts=salesbot_contexts
    )


if __name__ == "__main__":
    main()
