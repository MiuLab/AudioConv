"""
Modular Agent System for SalesBot Audio Dialogue
Full pipeline: Speech-to-Text (Whisper) → LLM reasoning (GPT) → Text-to-Speech (Sesame CSM)
"""

import torch
from openai import OpenAI
import whisper
from transformers import CsmForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from datasets import Dataset, Audio
import os
from typing import List, Dict, Optional
import json
import librosa
import soundfile as sf
import numpy as np


class AgentSystem:
    """
    Sales agent with full modular audio pipeline:
    1. Speech-to-text: Transcribe user audio input (Whisper)
    2. LLM reasoning: Generate agent response using SalesBot 2.0 prompts (GPT)
    3. Text-to-speech: Convert response to audio using Sesame CSM
    """

    def __init__(
        self,
        model_id: str = "sesame/csm-1b",
        whisper_model: str = "large-v2",
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        """
        Initialize Agent System with full audio pipeline

        Args:
            model_id: Sesame CSM model ID for TTS
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run models on
            openai_api_key: OpenAI API key for GPT
            openai_model: GPT model to use
        """
        # Setup device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load Whisper for speech-to-text
        print(f"Loading Whisper model ({whisper_model}) for STT...")
        self.whisper = whisper.load_model(whisper_model, device=self.device)

        # Load Sesame CSM for text-to-speech
        print(f"Loading Sesame CSM model on {self.device} for TTS...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id,
            device_map=self.device
        )

        # Setup OpenAI for response generation
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.openai_model = openai_model

        # Agent voice (speaker ID)
        self.speaker_id = "1"  # Default Sesame speaker ID

        # Dialogue state
        self.dialogue_phase = "chitchat"  # chitchat, transition, tod
        self.intent = None

        print(f"✓ Agent system initialized (using Sesame default speaker ID: {self.speaker_id})")

    def load_voice_profile(self, speaker_id: str = "1"):
        """
        Set agent speaker ID for Sesame CSM

        Args:
            speaker_id: Sesame speaker ID (e.g., "1", "2", etc.)
        """
        self.speaker_id = speaker_id
        print(f"Agent speaker ID set to: {speaker_id}")

    def speech_to_text(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            result = self.whisper.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def generate_agent_response(
        self,
        user_utterance: str,
        dialogue_history: List[Dict[str, str]],
        salesbot_context: Optional[Dict] = None
    ) -> str:
        """
        Generate agent response using GPT with SalesBot 2.0 prompts

        Args:
            user_utterance: User's transcribed utterance
            dialogue_history: Previous dialogue turns
            salesbot_context: SalesBot context (intents, products, etc.)

        Returns:
            Generated agent response text
        """
        messages = []

        # System prompt for sales agent
        system_prompt = """You are a friendly sales assistant. Your goal is to:
1. Build rapport through natural chitchat
2. Gradually understand the user's needs and interests
3. Smoothly transition to helping them with their intent
4. Provide helpful recommendations and complete tasks

Important output requirements (your reply will be fed directly to a TTS engine):
- Output plain conversational text only (no markdown, no code blocks, no lists, no emojis, no hashtags).
- Do not include stage directions or bracketed annotations (e.g., [laughs], *smiles*).
- Do not wrap the whole reply in quotes. Avoid backticks and special formatting symbols.
- Keep sentences natural and concise, with standard punctuation only.

Be conversational, empathetic, and helpful. Don't be too pushy or aggressive."""

        if salesbot_context:
            system_prompt += f"\n\nContext: {json.dumps(salesbot_context)}"

        messages.append({"role": "system", "content": system_prompt})

        # Add dialogue history
        for turn in dialogue_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["content"]})

        # Add current user utterance
        messages.append({"role": "user", "content": user_utterance})

        # Call GPT API
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.8,
                max_tokens=200
            )
            agent_text = response.choices[0].message.content.strip()
            agent_text = self._sanitize_for_tts(agent_text)
            return agent_text
        except Exception as e:
            print(f"Error generating agent response: {e}")
            return "I understand. How else can I help you?"

    def text_to_speech(self, text: str, output_path: str) -> str:
        """
        Convert agent response text to speech using Sesame CSM with default voice

        Args:
            text: Agent response text
            output_path: Output audio file path

        Returns:
            Path to generated audio file
        """
        # Use configured Sesame speaker ID (no voice profile cloning needed)
        conversation = [
            {"role": self.speaker_id, "content": [{"type": "text", "text": text}]},
        ]

        # Generate audio
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        # Generate with sufficient max_new_tokens for long text
        audio = self.model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=2048
        )
        self.processor.save_audio(audio, output_path)

        # Trim silence from start and end
        self._trim_silence(output_path)

        return output_path

    def _trim_silence(self, audio_path: str, top_db: int = 30):
        """
        Trim silence from the start and end of audio file

        Args:
            audio_path: Path to audio file to trim
            top_db: Threshold in decibels below reference to consider as silence
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            sf.write(audio_path, trimmed_audio, sr)
        except Exception as e:
            print(f"Warning: Could not trim silence from {audio_path}: {e}")

    def process_turn(
        self,
        user_audio_path: str,
        output_audio_path: str,
        dialogue_history: List[Dict[str, str]],
        salesbot_context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Process complete agent turn: STT → LLM → TTS

        Args:
            user_audio_path: Path to user audio input
            output_audio_path: Path to save agent audio output
            dialogue_history: Previous dialogue turns
            salesbot_context: SalesBot context

        Returns:
            Dict with "text", "audio_path", "transcription" keys
        """
        # Step 1: Speech-to-text
        print("Transcribing user audio...")
        user_text = self.speech_to_text(user_audio_path)
        print(f"User said: {user_text}")

        # Step 2: Generate agent response
        print("Generating agent response...")
        agent_text = self.generate_agent_response(
            user_text,
            dialogue_history,
            salesbot_context
        )
        print(f"Agent responds: {agent_text}")

        # Step 3: Text-to-speech
        print("Converting to speech...")
        audio_path = self.text_to_speech(agent_text, output_audio_path)

        return {
            "text": agent_text,
            "audio_path": audio_path,
            "transcription": user_text,
            "role": "agent"
        }

    def _sanitize_for_tts(self, text: str) -> str:
        """
        Make the text safe for direct TTS consumption.
        - Remove common markdown/code formatting characters
        - Strip surrounding quotes/backticks
        - Collapse whitespace
        """
        if not isinstance(text, str):
            return text
        import re

        # Remove code fences and backticks
        text = text.replace("```", " ")
        text = text.replace("`", " ")

        # Remove common markdown bullets/symbols
        text = text.replace("*", " ").replace("#", " ").replace("_", " ")

        # Remove bracketed stage directions like [laughs] or (sighs)
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"\([^\)]+\)", " ", text)

        # Strip surrounding quotes
        text = text.strip().strip('"').strip("'")

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
