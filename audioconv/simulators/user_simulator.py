"""
User Simulator: GPT-based text generation + Sesame CSM text-to-speech
Generates user responses in text, then converts to speech using Sesame CSM with voice profiles
"""

import torch
from openai import OpenAI
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import Dataset, Audio
import os
from typing import List, Dict, Optional
import json
import librosa
import soundfile as sf
import numpy as np


class UserSimulator:
    """
    Simulates user behavior in sales dialogues:
    1. GPT generates user text responses based on dialogue context
    2. Sesame CSM converts text to speech with voice profile (optional voice cloning)
    """

    def __init__(
        self,
        model_id: str = "sesame/csm-1b",
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        offload_to_cpu: bool = True
    ):
        """
        Initialize User Simulator with GPT and Sesame CSM

        Args:
            model_id: Sesame CSM model ID from HuggingFace
            device: Device to run Sesame CSM on (cuda:0, cpu, etc.)
            openai_api_key: OpenAI API key for GPT
            openai_model: GPT model to use (gpt-4, gpt-4o-mini, etc.)
            offload_to_cpu: Whether to move model to CPU after generation to save VRAM
        """
        # Setup device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.offload_to_cpu = offload_to_cpu and torch.cuda.is_available()

        # Load Sesame CSM for text-to-speech
        print(f"Loading Sesame CSM model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id,
            device_map=self.device
        )

        # Setup OpenAI for text generation
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.openai_model = openai_model

        # Voice profile context (audio samples for cloning)
        self.voice_profile = None
        self.speaker_id = "0"  # Default speaker ID

    def load_voice_profile(self, audio_files: List[str], texts: List[str], profile_name: str):
        """
        Load voice profile from audio samples for voice cloning

        Args:
            audio_files: List of audio file paths for voice profile
            texts: List of corresponding text transcriptions
            profile_name: Name identifier for this voice profile
        """
        # Load audio dataset for voice profile
        audio_dataset = Dataset.from_dict({
            "audio": audio_files,
            "text": texts,
            "profile": [profile_name] * len(audio_files)
        }).cast_column("audio", Audio(sampling_rate=24000))

        # Store voice profile context
        self.voice_profile = {
            "dataset": audio_dataset,
            "name": profile_name,
            "audio_files": audio_files,
            "texts": texts
        }

        print(f"Loaded voice profile: {profile_name} with {len(audio_files)} samples")

    def generate_user_response(
        self,
        dialogue_history: List[Dict[str, str]],
        user_persona: Optional[str] = None,
        task_context: Optional[Dict] = None
    ) -> str:
        """
        Generate user text response using GPT based on dialogue context

        Args:
            dialogue_history: List of dialogue turns [{"role": "user"/"agent", "content": "..."}]
            user_persona: Optional user persona description
            task_context: Optional task-specific context (intent, preferences, etc.)

        Returns:
            Generated user response text
        """
        messages = []

        # System prompt
        system_prompt = (
            "You are simulating the USER (customer) in a sales dialogue. "
            "Always speak as the user in first person (I/me). "
            "Do not act like the agent or provide assistance, recommendations, or instructions to the agent. "
            "Avoid agent-like phrases such as 'I can give you the best recommendations' or 'Let me know your city so I can suggest options.' "
            "Focus on expressing preferences, goals, questions, or reactions as a customer. "
            "Output plain conversational text only (no markdown, speaker tags, or stage directions). "
            "Do not prefix with 'User:' or 'Agent:'. "
            "Your reply will be passed directly to a TTS engine."
        )
        if user_persona:
            system_prompt += f" User persona: {user_persona}."
        if task_context:
            system_prompt += f" Task context: {json.dumps(task_context)}."

        messages.append({"role": "system", "content": system_prompt})

        # Add dialogue history
        for turn in dialogue_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["content"]})

        # Call GPT API
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.8,
                max_tokens=150
            )
            user_text = response.choices[0].message.content.strip()
            user_text = self._sanitize_for_tts(user_text)
            return user_text
        except Exception as e:
            print(f"Error generating user response with GPT: {e}")
            return "I see. Could you tell me more?"

    def text_to_speech(self, text: str, output_path: str) -> str:
        """
        Convert text to speech using Sesame CSM with loaded voice profile

        Args:
            text: Text to convert to speech
            output_path: Output audio file path

        Returns:
            Path to generated audio file
        """
        # Move model to GPU if offloaded
        if self.offload_to_cpu:
            print("Moving CSM model to GPU...")
            self.model = self.model.to(self.device)
            torch.cuda.empty_cache()

        conversation = []

        # Add voice profile context if available
        if self.voice_profile:
            dataset = self.voice_profile["dataset"]
            for row in dataset:
                conversation.append({
                    "role": self.speaker_id,
                    "content": [
                        {"type": "text", "text": row["text"]},
                        {"type": "audio", "path": row["audio"]["array"]}
                    ]
                })

        # Add text prompt to generate
        conversation.append({
            "role": self.speaker_id,
            "content": [{"type": "text", "text": text}]
        })

        # Generate audio
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        audio = self.model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=2048
        )
        self.processor.save_audio(audio, output_path)

        # Move model to CPU to free VRAM for agent model
        if self.offload_to_cpu:
            print("Offloading CSM model to CPU to save VRAM...")
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        # Trim silence from start and end
        self._trim_silence(output_path)

        return output_path

    def simulate_turn(
        self,
        dialogue_history: List[Dict[str, str]],
        output_audio_path: str,
        user_persona: Optional[str] = None,
        task_context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Simulate complete user turn: generate text + convert to speech

        Args:
            dialogue_history: List of dialogue turns
            output_audio_path: Path to save generated audio
            user_persona: Optional user persona
            task_context: Optional task context

        Returns:
            Dict with "text" and "audio_path" keys
        """
        # Step 1: Generate user text response with GPT
        user_text = self.generate_user_response(
            dialogue_history,
            user_persona,
            task_context
        )

        # Step 2: Convert to speech with Sesame CSM
        audio_path = self.text_to_speech(user_text, output_audio_path)

        return {
            "text": user_text,
            "audio_path": audio_path,
            "role": "user"
        }

    def _trim_silence(self, audio_path: str, top_db: int = 30):
        """Trim silence from the start and end of audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            sf.write(audio_path, trimmed_audio, sr)
        except Exception as e:
            print(f"Warning: Could not trim silence from {audio_path}: {e}")

    def _sanitize_for_tts(self, text: str) -> str:
        """
        Make the text safe for direct TTS and enforce user role style.
        - Remove markdown/code formatting and speaker tags
        - Strip surrounding quotes/backticks
        - Collapse whitespace
        """
        if not isinstance(text, str):
            return text
        import re

        # Remove speaker tags if present
        text = re.sub(r"^(User|Agent)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)

        # Remove code fences and backticks
        text = text.replace("```", " ")
        text = text.replace("`", " ")

        # Remove common markdown symbols
        text = text.replace("*", " ").replace("#", " ").replace("_", " ")

        # Remove bracketed stage directions
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"\([^\)]+\)", " ", text)

        # Strip surrounding quotes
        text = text.strip().strip('"').strip("'")

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
