"""
Agent System with MiMo-Audio-7B-Instruct
End-to-end multimodal agent with native speech generation (audio-in → audio-out)
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import os

from audioconv.llms.mimo import MiMoAudioChat


class AgentSystemMiMo:
    """
    Sales agent backed by Xiaomi MiMo-Audio-7B-Instruct for end-to-end audio dialogue
    """

    def __init__(
        self,
        model_id: str = "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        tokenizer_id: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
        device: str = "cuda:0",
        prompt_speech: Optional[str] = None,
        max_history_turns: int = 6,
    ) -> None:
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.device = device
        self.prompt_speech = prompt_speech
        self.max_history_turns = max(0, int(max_history_turns))

        # MiMo model wrapper (loads on GPU 0 by default)
        self.llm = MiMoAudioChat(model_id=model_id, tokenizer_id=tokenizer_id, device=device)

        # Default system prompt for SalesBot behavior
        self.system_prompt = (
            "You are a helpful, friendly sales assistant. "
            "Have natural chitchat, understand the user's needs, and offer helpful, concrete suggestions."
        )

        # SalesBot task guidance appended per turn
        self.salesbot_instructions = (
            "Be conversational and empathetic. If possible, suggest attractions, restaurants, movies, hotels, or events based on context."
        )

        # Whether to include audio files in dialogue history (not just text)
        self.use_audio_history = False  # Default: text-only for memory safety

    def process_turn(
        self,
        user_audio_path: str,
        output_audio_path: str,
        dialogue_history: List[Dict[str, str]],
        salesbot_context: Optional[Dict] = None,
        use_audio_history: Optional[bool] = None
    ) -> Dict[str, Optional[str]]:
        """
        Complete agent turn: audio in → generate response → audio out

        Args:
            user_audio_path: Path to user audio input
            output_audio_path: Path to save agent audio output
            dialogue_history: Previous dialogue turns (with 'content' and optional 'audio_path' keys)
            salesbot_context: SalesBot context
            use_audio_history: Whether to include audio files from history (overrides default)
                              True = Include audio files for audio-native processing
                              False = Text-only (memory efficient)
                              None = Use instance default (self.use_audio_history)

        Returns a dict with keys: text, audio_path, transcription, role, audio_native
        """
        conversations = []

        # System
        conversations.append({
            "role": "system",
            "parts": [{"type": "text", "value": self.system_prompt}]
        })

        use_audio = use_audio_history if use_audio_history is not None else self.use_audio_history

        history = dialogue_history[-self.max_history_turns:] if self.max_history_turns > 0 else []
        for turn in history:
            role = "assistant" if turn.get("role") == "agent" else "user"
            text_val = turn.get("content") or ""
            audio_path = turn.get("audio_path", "")

            parts_list = []
            if use_audio and audio_path and os.path.exists(audio_path):
                parts_list.append({"type": "audio", "value": audio_path})
            if text_val:
                parts_list.append({"type": "text", "value": text_val})
            if parts_list:
                conversations.append({"role": role, "parts": parts_list})

        # Compose per-turn instruction
        instruction_text = self.salesbot_instructions
        if salesbot_context:
            instruction_text += f"\n\nContext: {json.dumps(salesbot_context)}"
        instruction_text += "\n\nRespond naturally as a helpful sales agent to the audio message."

        # Current user audio
        conversations.append({
            "role": "user",
            "parts": [
                {"type": "audio", "value": user_audio_path},
                {"type": "text", "value": instruction_text},
            ],
        })

        # Run MiMo
        text, audio_path = self.llm(
            conversations,
            output_audio_path=output_audio_path,
            system_prompt=self.system_prompt,
            prompt_speech=self.prompt_speech,
        )

        return {
            "text": text,
            "audio_path": audio_path,
            "transcription": "Audio processed internally",
            "role": "agent",
            "audio_native": True if audio_path else False,
        }
