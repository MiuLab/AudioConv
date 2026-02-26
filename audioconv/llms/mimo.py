import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from packaging.version import parse as vparse


def _ensure_mimo_src_on_path():
    """Ensure MiMo-Audio is importable.

    We try to make `src.mimo_audio.mimo_audio` importable by appending the
    MiMo-Audio repository root to `sys.path`. As a fallback, we also append the
    `src` directory so importing `mimo_audio.mimo_audio` works.
    """
    here = Path(__file__).resolve()
    # AudioConv/audioconv/llms/mimo.py -> parents[3] = project root containing MiMo-Audio/
    repo_root = here.parents[3]
    mimo_root = repo_root / "MiMo-Audio"
    mimo_src = mimo_root / "src"
    # Prefer MiMo-Audio root to support `import src.mimo_audio...`
    if mimo_root.exists():
        mimo_root_str = str(mimo_root)
        if mimo_root_str not in sys.path:
            sys.path.insert(0, mimo_root_str)
    # Also add src for `import mimo_audio...` fallback
    if mimo_src.exists():
        mimo_src_str = str(mimo_src)
        if mimo_src_str not in sys.path:
            sys.path.insert(0, mimo_src_str)


_ensure_mimo_src_on_path()

# Import after path injection
try:
    try:
        from src.mimo_audio.mimo_audio import MimoAudio  # type: ignore
    except Exception:
        from mimo_audio.mimo_audio import MimoAudio  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import MiMo-Audio. Ensure MiMo-Audio/ is present and readable."
    ) from e

# Note: We patched MiMo to inherit GenerationMixin, so modern transformers (>=4.50)
# are supported. No version guard needed here.


class MiMoAudioChat:
    """
    Thin wrapper around XiaomiMiMo/MiMo-Audio-7B-Instruct for audio-to-audio chat.

    - Loads model on CUDA GPU 0 by default (device="cuda:0").
    - Accepts the repo's conversation format (list of {role, parts|content}).
    - Produces an audio reply and returns both text and audio path.

    Example conversation item formats supported:
      {"role": "system", "parts": [{"type": "text", "value": "..."}]}
      {"role": "user",   "parts": [{"type": "audio", "value": "path.wav"}]}
      {"role": "assistant", "parts": [
          {"type": "text",  "value": "hello"},
          {"type": "audio", "value": "path.wav"}
      ]}

    or the Qwen-style content list used by agents/qwen_omni_agent.py:
      {"role": "user", "content": [
          {"type": "audio", "audio": "path.wav"},
          {"type": "text",  "text":  "instruction"}
      ]}
    """

    def __init__(
        self,
        model_id: str = "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        tokenizer_id: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
        device: str = "cuda:0",
        quantized: bool = True,
    ) -> None:
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.device = device

        # Instantiate MiMo-Audio on the requested device (GPU 0 by default)
        start = time.time()
        self.model = MimoAudio(model_id, tokenizer_id, device=device, quantized=quantized)
        self.load_seconds = time.time() - start

    def __str__(self) -> str:
        return self.model_id.replace("/", "-")

    # ------------------------------
    # Public API
    # ------------------------------
    def __call__(
        self,
        conversations: List[Dict[str, Any]],
        output_audio_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_speech: Optional[str] = None,
        thinking: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """
        Run multi-turn audio-to-audio dialogue and return (text, audio_path).

        Args:
            conversations: List of {role, parts|content} dicts.
            output_audio_path: Where to save the assistant's reply audio (WAV). If None, saves under test_outputs/.
            system_prompt: Optional global instruction prompt.
            prompt_speech: Optional reference voice audio path for conditioning the assistant voice.
            thinking: If True, enables thinking tokens for text channel where applicable.

        Returns:
            (assistant_text, saved_audio_path)
        """
        message_list, sys_prompt, user_instruction = self._to_mimo_messages(conversations)
        # Allow explicit system_prompt to override conversation-derived
        if system_prompt:
            sys_prompt = system_prompt

        # Check if there's any audio in the message_list
        has_audio = self._has_audio_content(message_list)

        # Route to appropriate method based on audio presence
        if not has_audio:
            # Text-only dialogue: use text_dialogue_sft_multiturn
            # Convert message_list to text-only format
            text_message_list = self._convert_to_text_only_messages(message_list, conversations)

            assistant_text = self.model.text_dialogue_sft_multiturn(
                text_message_list,
                thinking=thinking,
            )
            saved_audio = None
        else:
            # Audio dialogue: use spoken_dialogue_sft_multiturn
            # Where to save audio
            if output_audio_path is None:
                out_dir = Path("test_outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time() * 1000)
                output_audio_path = str(out_dir / f"mimo_reply_{ts}.wav")

            assistant_text = self.model.spoken_dialogue_sft_multiturn(
                message_list,
                output_audio_path=output_audio_path,
                system_prompt=sys_prompt,
                prompt_speech=prompt_speech,
            )

            # If the model didn't actually write audio, keep path only if file exists
            saved_audio = output_audio_path if os.path.exists(output_audio_path) else None

        # MiMo returns text including special markers; keep only natural assistant text
        if "<|eot|>" in assistant_text:
            assistant_text = assistant_text.split("<|eot|>")[0]
        assistant_text = assistant_text.replace(".....", "").strip().split('Assistant')[0].strip().replace('</think>','').replace('<think>','')

        return assistant_text, saved_audio

    def clear_history(self) -> None:
        """Clear MiMo conversation history state."""
        self.model.clear_history()

    # ------------------------------
    # Helpers
    # ------------------------------
    def _to_mimo_messages(
        self, conversations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Convert repo conversation structure to MiMo `message_list`.

        Returns:
            (message_list, system_prompt, user_instruction_text)
        """
        message_list: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        user_instruction_text: Optional[str] = None

        for turn in conversations:
            role = turn.get("role")
            parts = turn.get("parts")
            content = turn.get("content")

            # Normalize to a list of unit parts with possible keys:
            #  - {type: 'text', value: str} or {type: 'text', text: str}
            #  - {type: 'audio', value: path} or {type: 'audio', audio: path}
            norm_parts: List[Dict[str, Any]] = []
            if isinstance(parts, list):
                norm_parts = parts
            elif isinstance(content, list):
                # Convert to unified keys
                for p in content:
                    if not isinstance(p, dict) or "type" not in p:
                        continue
                    if p["type"] == "text":
                        norm_parts.append({"type": "text", "value": p.get("text")})
                    elif p["type"] == "audio":
                        norm_parts.append({"type": "audio", "value": p.get("audio")})
            elif isinstance(content, str):
                # assistant/user text as plain string
                norm_parts = [{"type": "text", "value": content}]

            if role == "system":
                # Use the first text piece as system prompt if present
                for p in norm_parts:
                    if p.get("type") == "text" and p.get("value"):
                        system_prompt = p["value"]
                        break
                continue

            if role == "user":
                audio_path = None
                text_val = None
                for p in norm_parts:
                    if p.get("type") == "audio" and p.get("value"):
                        audio_path = p["value"]
                    elif p.get("type") == "text" and p.get("value"):
                        text_val = p["value"]
                # Capture instruction text (not inserted into message_list)
                if text_val and not user_instruction_text:
                    user_instruction_text = text_val
                # For spoken dialogue, MiMo expects user content to be audio only
                if audio_path:
                    message_list.append({"role": "user", "content": audio_path})
                # Skip text-only user turns to avoid format mismatches
                continue

            if role == "assistant":
                audio_path = None
                text_val = None
                for p in norm_parts:
                    if p.get("type") == "audio" and p.get("value"):
                        audio_path = p["value"]
                    elif p.get("type") == "text" and p.get("value"):
                        text_val = p["value"]
                # For spoken dialogue multi-turn, MiMo expects BOTH text and audio for assistant history
                if audio_path and text_val:
                    message_list.append({"role": "assistant", "content": {"text": text_val, "audio": audio_path}})
                # Else, skip assistant-only text or audio to avoid schema errors
                continue

        return message_list, system_prompt, user_instruction_text

    def _has_audio_content(self, message_list: List[Dict[str, Any]]) -> bool:
        """
        Check if any message in the list contains audio content.

        Args:
            message_list: List of messages in MiMo format.

        Returns:
            True if any message contains audio, False otherwise.
        """
        for msg in message_list:
            content = msg.get("content")
            if isinstance(content, str) and os.path.exists(content):
                # User message with audio path
                return True
            elif isinstance(content, dict) and "audio" in content:
                # Assistant message with audio
                return True
        return False

    def _convert_to_text_only_messages(
        self, message_list: List[Dict[str, Any]], conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert message_list to text-only format for text_dialogue_sft_multiturn.

        Args:
            message_list: MiMo format message list (may be incomplete for text-only).
            conversations: Original conversation structure.

        Returns:
            List of {role: str, content: str} for text dialogue.
        """
        text_messages: List[Dict[str, str]] = []

        for turn in conversations:
            role = turn.get("role")
            if role == "system":
                # System messages are handled separately via system_prompt
                continue

            parts = turn.get("parts")
            content = turn.get("content")

            # Extract text from parts or content
            text_val = None
            if isinstance(parts, list):
                for p in parts:
                    if p.get("type") == "text" and p.get("value"):
                        text_val = p["value"]
                        break
            elif isinstance(content, list):
                for p in content:
                    if p.get("type") == "text" and p.get("text"):
                        text_val = p["text"]
                        break
            elif isinstance(content, str):
                text_val = content

            # Add user and assistant text messages
            if role in ("user", "assistant") and text_val:
                text_messages.append({"role": role, "content": text_val})

        return text_messages
