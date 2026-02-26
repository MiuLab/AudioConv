# AudioConv: The Context Trap

**Code for:** *The Context Trap: Why End-to-End Audio Language Models Fail Multi-turn Dialogues*
IWSDS 2026 | NTU MiuLab

---

## Overview

This repository provides the complete pipeline for evaluating E2E AudioLMs vs. modular (ASR→LLM→TTS) systems on multi-turn sales dialogue. The paper shows that E2E models exhibit **severe per-turn dialogue degradation** — not because of audio generation quality, but due to **context maintenance failures** (topic drift, repetition, memory loss).

### Key Finding

E2E audio models produce natural-sounding speech but fail to maintain coherent dialogue context across turns. The problem is **dialogue modeling**, not audio quality.

### Pipeline

```
JSON Dataset  ──(Step 1)──►  User Audio  ──(Step 2/3)──►  Agent Transcripts  ──(Steps 4-6)──►  Scores
              Sesame CSM TTS              Audio/Text Agents                     LLM-as-Judge
```

---

## Models Compared

| System | Type | ASR | LLM | TTS |
|--------|------|-----|-----|-----|
| Modular (Sesame) | Pipeline | Whisper-large-v2 | GPT-4o-mini | Sesame CSM-1B |
| Qwen2.5-Omni-3B | E2E | — | — | — |
| Qwen2.5-Omni-7B | E2E | — | — | — |
| MiMo-Audio-7B | E2E | — | — | — |

---

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For MiMo-Audio (E2E model), clone alongside this repo:
git clone https://github.com/XiaomiMiMo/MiMo-Audio.git ../MiMo-Audio
pip install -r ../MiMo-Audio/requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"        # Required for modular agent + eval
export GEMINI_API_KEY="your-gemini-key"         # Required for audio quality eval
export GCP_PROJECT_NAME="your-gcp-project"      # Required for Gemini (Vertex AI)
```

---

## Data

Download the MSGD dataset (500 dialogues) and place it at `data/MSGD_dataset_final.json`.

See [data/README.md](data/README.md) for details.

---

## Quickstart

### Step 1 — Generate User Audio

Convert MSGD text dialogues to user speech using Sesame CSM TTS:

```bash
python scripts/1_prepare_user_audio.py \
  --dataset data/MSGD_dataset_final.json \
  --output_dir data/user_audio \
  --num_dialogues 500
```

### Step 2 — Run Audio Agents

Run each agent system on the fixed user audio inputs:

```bash
# Modular baseline (Whisper + GPT-4o-mini + Sesame CSM)
python scripts/2_run_audio_agents.py \
  --agent_type sesame \
  --openai_model gpt-4o-mini \
  --output_dir dialogue_outputs

# Qwen2.5-Omni-7B (E2E)
python scripts/2_run_audio_agents.py \
  --agent_type qwen_omni_7b \
  --speaker Chelsie \
  --output_dir dialogue_outputs

# MiMo-Audio-7B (E2E)
python scripts/2_run_audio_agents.py \
  --agent_type mimo \
  --output_dir dialogue_outputs
```

### Step 3 — Run Text-Only Agents (Baseline)

```bash
# GPT-4o-mini text-only baseline
python scripts/3_run_text_agents.py \
  --agent_type gpt4o_mini \
  --dataset data/MSGD_dataset_final.json \
  --output_dir dialogue_outputs_text_only
```

### Step 4 — Evaluate Dialogue Quality

Score naturalness and consistency (0–100) using LLM-as-Judge:

```bash
python scripts/4_eval_dialogue_quality.py \
  --transcript_dir dialogue_outputs/agent_sesame_gpt-4o-mini_speaker1_fixed_eval \
  --output_file eval_results/dialogue_quality_sesame.json \
  --model gpt-4o-mini

# Audio agents (fill missing user text from dataset)
python scripts/4_eval_dialogue_quality.py \
  --transcript_dir dialogue_outputs/agent_omni_7b-chelsie_fixed_eval \
  --msgd_dataset data/MSGD_dataset_final.json \
  --output_file eval_results/dialogue_quality_omni7b.json \
  --model gemini-2.5-flash
```

### Step 5 — Evaluate Audio Quality

Score audio perceptual quality using Gemini (naturalness, clarity, consistency, etc.):

```bash
python scripts/5_eval_audio_quality.py \
  --audio_dir dialogue_outputs/agent_sesame_gpt-4o-mini_speaker1_fixed_eval \
  --output eval_results/audio_quality_sesame.json \
  --model gemini-2.5-flash
```

### Step 6 — Analyze Failure Patterns

Classify dialogue failures into 5 categories (generic, topic drift, repetition, misunderstanding, memory):

```bash
python scripts/6_analyze_failures.py \
  --transcript_dir dialogue_outputs/agent_mimo-audio_fixed_eval \
  --msgd_dataset data/MSGD_dataset_final.json \
  --output_file eval_results/failure_patterns_mimo.json \
  --model gpt-4o-mini
```

---

## Repository Structure

```
AudioConv/
├── README.md
├── requirements.txt
│
├── audioconv/                    # Installable core library
│   ├── agents/
│   │   ├── modular_agent.py      # Whisper + GPT + Sesame CSM
│   │   ├── qwen_omni_agent.py    # Qwen2.5-Omni (E2E)
│   │   └── mimo_agent.py         # MiMo-Audio-7B (E2E)
│   ├── simulators/
│   │   └── user_simulator.py     # GPT text gen + Sesame CSM TTS
│   ├── llms/
│   │   ├── mimo.py               # MiMo-Audio wrapper
│   │   ├── gemini.py             # Gemini API wrapper
│   │   ├── utils.py              # LLM factory + retry logic
│   │   ├── qwen_omni_utils.py    # Qwen Omni audio/vision utilities
│   │   ├── audio_process.py      # Audio preprocessing
│   │   └── vision_process.py     # Vision preprocessing
│   └── prompt_templates.py       # SalesBot prompt templates
│
├── eval/
│   ├── dialogue_quality.py       # LLM-as-Judge (naturalness + consistency)
│   ├── audio_quality.py          # Gemini perceptual audio evaluator
│   └── failure_analysis.py       # 5-category failure pattern classifier
│
├── scripts/                      # Pipeline entry points (run in order)
│   ├── 1_prepare_user_audio.py
│   ├── 2_run_audio_agents.py
│   ├── 3_run_text_agents.py
│   ├── 4_eval_dialogue_quality.py
│   ├── 5_eval_audio_quality.py
│   └── 6_analyze_failures.py
│
└── data/
    ├── README.md                 # Data download instructions
    └── user_audio/               # Generated by Step 1
```

---

## Citation

```bibtex
@inproceedings{audioconv2026,
  title     = {The Context Trap: Why End-to-End Audio Language Models Fail Multi-turn Dialogues},
  booktitle = {IWSDS 2026},
  year      = {2026},
}

@inproceedings{chang2024injecting,
  title     = {Injecting Salesperson's Dialogue Strategies in Large Language Models with Chain-of-Thought Reasoning},
  author    = {Wen-Yu Chang and Yun-Nung Chen},
  booktitle = {Findings of ACL 2024},
  pages     = {3798--3812},
  year      = {2024},
}

@article{chang2023salesbot,
  title   = {SalesBot 2.0: A Human-Like Intent-Guided Chit-Chat Dataset},
  author  = {Chang, Wen-Yu and Chen, Yun-Nung},
  journal = {arXiv preprint arXiv:2308.14266},
  year    = {2023},
}
```
