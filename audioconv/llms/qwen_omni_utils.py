"""
Utility functions for Qwen2.5-Omni multimodal processing
"""

import torch
import torchaudio
from PIL import Image
from typing import List, Dict, Tuple, Optional
import os
from audioconv.llms.audio_process import process_audio_info
from audioconv.llms.vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)


def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (Hz)

    Returns:
        Audio tensor
    """
    audio_array, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_array = resampler(audio_array)

    # Convert to mono if stereo
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)

    return audio_array


def save_audio(audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """
    Save audio tensor to file

    Args:
        audio_tensor: Audio tensor to save
        output_path: Output file path
        sample_rate: Sample rate for output
    """
    torchaudio.save(output_path, audio_tensor.cpu(), sample_rate)
