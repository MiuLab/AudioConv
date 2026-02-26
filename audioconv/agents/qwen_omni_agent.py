"""
Agent System with Qwen2.5-Omni
End-to-end multimodal agent with native speech generation
Replaces Whisper STT + GPT + Sesame TTS pipeline
"""

import torch
import sys
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from typing import List, Dict, Optional
import json
import os
import librosa
import soundfile as sf
import numpy as np

from audioconv.llms.qwen_omni_utils import process_mm_info, save_audio


class AgentSystemOmni:
    """
    Sales agent with Qwen2.5-Omni for end-to-end audio dialogue
    Single model handles: Audio input → Reasoning → Audio output
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Omni-3B",
        device: Optional[str] = None,
        quantized: bool = False,
        enable_talker: bool = True,
        speaker: str = "Ethan",
        max_history_turns: int = 6,
        use_gradient_checkpointing: bool = True
    ):
        """
        Initialize Agent System with Qwen2.5-Omni

        Args:
            model_id: Qwen2.5-Omni model ID (Qwen/Qwen2.5-Omni-3B or Qwen/Qwen2.5-Omni-7B)
            device: Device to run model on
            quantized: Whether to use 4-bit quantization
            enable_talker: Enable audio output generation
            speaker: Voice type ("Ethan" for male, "Chelsie" for female)
            max_history_turns: Number of past messages to include
            use_gradient_checkpointing: Enable gradient checkpointing to reduce VRAM
        """
        self.speaker = speaker
        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auto-enable quantization for 7B model
        if '7B' in model_id:
            quantized = True
            print("Auto-enabling 4-bit quantization for 7B model")

        # Setup quantization
        quantization_config = None
        if quantized:
            print("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )

        # Load Qwen2.5-Omni
        print(f"Loading Qwen2.5-Omni model on {self.device}...")
        try:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_id,
                device_map=self.device,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2",
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

            # Configure talker (audio generation)
            if not enable_talker or not hasattr(self.model, 'talker'):
                self.model.disable_talker()
                self.audio_enabled = False
                print("⚠ Audio output disabled (talker not available or disabled)")
            else:
                self.audio_enabled = True
                print("✓ Audio output enabled")

            print("✓ Loaded Qwen2.5-Omni model")

        except Exception as e:
            print(f"Error loading Qwen2.5-Omni: {e}")
            raise

        # Qwen Omni default system prompt (required for audio output)
        self.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

        # SalesBot 2.0 instructions (added as context instead of system prompt)
        self.salesbot_instructions = """You are helping customers find what they need.
Your goal is to:
1. Engage in natural chitchat to build rapport
2. Smoothly transition to understanding customer needs
3. Help customers with task-oriented requests (finding attractions, restaurants, movies, music, hotels, events, transportation, flights)

Be conversational, empathetic, and helpful. Make reasonable suggestions based on context."""

        self.max_history_turns = max(0, int(max_history_turns))

        # Whether to include audio files in dialogue history (not just text)
        self.use_audio_history = False  # Default: text-only for memory safety

    def process_turn(
        self,
        user_audio_path: str,
        output_audio_path: str,
        dialogue_history: List[Dict[str, str]],
        salesbot_context: Optional[Dict] = None,
        use_audio_history: Optional[bool] = None
    ) -> Dict[str, str]:
        """
        Process complete agent turn: Listen to audio → Generate response → Output audio

        Args:
            user_audio_path: Path to user audio input
            output_audio_path: Path to save agent audio output
            dialogue_history: Previous dialogue turns (with 'content' and optional 'audio_path' keys)
            salesbot_context: SalesBot context
            use_audio_history: Whether to include audio files from history (overrides default)
                              True = Include audio files for audio-native processing
                              False = Text-only (memory efficient)
                              None = Use instance default (self.use_audio_history)

        Returns:
            Dict with "text", "audio_path", "transcription" keys
        """
        # Build instruction text
        instruction_text = self.salesbot_instructions
        if salesbot_context:
            instruction_text += f"\n\nContext: {json.dumps(salesbot_context)}"
        instruction_text += "\n\nRespond naturally as a helpful sales agent to the audio message."

        use_audio = use_audio_history if use_audio_history is not None else self.use_audio_history

        # Progressive retry strategy to handle OOM during audio generation
        attempt_settings = [
            {"max_new_tokens": 2048, "temperature": 0.7, "do_sample": True,  "history_turns": self.max_history_turns},
            {"max_new_tokens": 1024, "temperature": 0.7, "do_sample": True,  "history_turns": min(self.max_history_turns, 3)},
            {"max_new_tokens": 768,  "temperature": 0.3, "do_sample": False, "history_turns": min(self.max_history_turns, 2)},
            {"max_new_tokens": 512,  "temperature": 0.1, "do_sample": False, "history_turns": 1},
        ]

        last_exception = None
        for i, cfg in enumerate(attempt_settings, 1):
            try:
                # Build conversation per attempt with reduced history
                convo_try = []
                convo_try.append({
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })

                history_n = max(0, int(cfg["history_turns"]))
                history = dialogue_history[-history_n:] if history_n > 0 else []
                for turn in history:
                    role = "assistant" if turn.get("role") == "agent" else "user"
                    text_val = turn.get("content", "")
                    audio_path = turn.get("audio_path", "")

                    content_list = []
                    if use_audio and audio_path and os.path.exists(audio_path):
                        content_list.append({"type": "audio", "audio": audio_path})
                    if text_val:
                        content_list.append({"type": "text", "text": text_val})
                    if content_list:
                        convo_try.append({"role": role, "content": content_list})

                convo_try.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": user_audio_path},
                        {"type": "text", "text": instruction_text}
                    ]
                })

                # Apply chat template and preprocess
                text = self.processor.apply_chat_template(
                    convo_try,
                    tokenize=False,
                    add_generation_prompt=True
                )
                audios, images, videos = process_mm_info(convo_try, use_audio_in_video=False)
                inputs = self.processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False
                )
                inputs = inputs.to(self.model.device)

                # Generate response with configured voice
                print(f"Generating agent response (attempt {i}, speaker: {self.speaker}, max_new_tokens={cfg['max_new_tokens']}, history_turns={cfg['history_turns']})...")
                with torch.no_grad():
                    if self.audio_enabled:
                        text_ids, audio_output = self.model.generate(
                            **inputs,
                            max_new_tokens=cfg["max_new_tokens"],
                            temperature=cfg["temperature"],
                            do_sample=cfg["do_sample"],
                            return_audio=True,
                            use_audio_in_video=False,
                            speaker=self.speaker
                        )
                    else:
                        text_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=cfg["max_new_tokens"],
                            temperature=cfg["temperature"],
                            do_sample=cfg["do_sample"],
                            use_audio_in_video=False
                        )
                        audio_output = None

                # Clean up GPU cache after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                agent_text = self.processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                # Extract only assistant response
                if "assistant\n" in agent_text:
                    agent_text = agent_text.split("assistant\n")[-1].strip()

                print(f"Agent responds: {agent_text}")

                # Save audio output if available
                audio_path = None
                if audio_output is not None and self.audio_enabled:
                    print("Saving audio output...")
                    audio_np = audio_output.reshape(-1).detach().cpu().numpy()
                    sf.write(output_audio_path, audio_np, samplerate=24000, format="WAV")
                    self._trim_silence(output_audio_path)
                    audio_path = output_audio_path
                else:
                    print("⚠ No audio output generated in this attempt")

                return {
                    "text": agent_text,
                    "audio_path": audio_path,
                    "transcription": "Audio processed internally",
                    "role": "agent",
                    "audio_native": self.audio_enabled
                }

            except torch.cuda.OutOfMemoryError as e:
                last_exception = e
                print(f"CUDA OOM encountered on attempt {i}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    last_exception = e
                    print(f"Runtime OOM encountered on attempt {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                    continue
                else:
                    raise
            except Exception:
                raise

        # If all attempts failed due to OOM, return minimal text-only fallback
        print("All audio-generation attempts failed due to OOM; returning text-only fallback.")
        try:
            minimal_conv = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": instruction_text}]},
            ]
            text = self.processor.apply_chat_template(minimal_conv, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                text_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=False)
            agent_text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if "assistant\n" in agent_text:
                agent_text = agent_text.split("assistant\n")[-1].strip()
        except Exception:
            agent_text = "I'm here to help. How can I assist you?"

        return {
            "text": agent_text,
            "audio_path": None,
            "transcription": "OOM encountered; audio regeneration failed",
            "role": "agent",
            "audio_native": False
        }

    def _trim_silence(self, audio_path: str, top_db: int = 30):
        """Trim silence from the start and end of audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            sf.write(audio_path, trimmed_audio, sr)
        except Exception as e:
            print(f"Warning: Could not trim silence from {audio_path}: {e}")
