# Medical Domain Assistant via LLM Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sREK2luv_bLZWwq3EnwWHZ3RkwiFPAvf?usp=sharing)

A medical question-answering assistant built by fine-tuning **TinyLlama-1.1B-Chat** using **LoRA** (Low-Rank Adaptation) on medical flashcard data. The model provides accurate, concise answers to medical questions and is deployed through an interactive Gradio chat interface.

## Project Overview

| Component | Details |
|-----------|---------|
| **Domain** | Healthcare / Medical Education |
| **Base Model** | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **Dataset** | [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) (33,955 Q&A pairs, subsampled to 3,000) |
| **Fine-tuning** | LoRA via `peft` with 4-bit quantization (`bitsandbytes`) |
| **Evaluation** | BLEU, ROUGE-1/2/L, Perplexity |
| **Interface** | Gradio Chat UI |

## Dataset

The **medical_meadow_medical_flashcards** dataset contains 33,955 medical question-answer pairs derived from medical flashcards. It covers a broad range of topics including:
- Anatomy & Physiology
- Pharmacology & Drug Mechanisms
- Pathology & Disease Processes
- Clinical Medicine & Diagnostics
- Biochemistry & Molecular Biology

### Preprocessing
- Cleaned text (removed noise, normalized whitespace)
- Filtered out entries shorter than 10 characters or longer than 2,000 characters
- Subsampled to 3,000 high-quality examples
- Formatted into TinyLlama's chat template (`<|system|>`, `<|user|>`, `<|assistant|>`)
- Split into train (80%), validation (10%), and test (10%) sets

## Fine-Tuning Methodology

### LoRA Configuration
- **Quantization**: 4-bit NF4 with double quantization
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- **LoRA dropout**: 0.05
- **Optimizer**: paged AdamW 8-bit

### Hyperparameter Experiments

| Experiment | Learning Rate | Batch Size | Epochs | LoRA Rank | LoRA Alpha |
|-----------|--------------|------------|--------|-----------|------------|
| Exp 1 | 2e-4 | 4 | 3 | 16 | 32 |
| Exp 2 | 5e-5 | 2 | 3 | 16 | 32 |
| Exp 3 | 1e-4 | 4 | 2 | 8 | 16 |

All experiments use gradient accumulation to achieve an effective batch size of 8, with FP16 mixed precision training and a warmup ratio of 0.1.

## Performance Metrics

Metrics are computed on the test set after selecting the best experiment (lowest validation loss). The notebook contains full evaluation results including:

- **BLEU Score**: Measures n-gram precision against reference answers
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: Measures recall-oriented overlap
- **Perplexity**: Measures model confidence on test data (lower is better)

See the notebook for the complete metrics table and experiment comparison charts.

## Example Conversations

### Fine-Tuned Model vs. Base Model

**Q: What are the main symptoms of Type 2 diabetes?**

| Base Model | Fine-Tuned Model |
|-----------|-----------------|
| Generic, unfocused response about diabetes | Specific listing of symptoms: polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow wound healing |

**Q: Explain the mechanism of action of aspirin.**

| Base Model | Fine-Tuned Model |
|-----------|-----------------|
| Vague description | Detailed explanation of COX-1/COX-2 inhibition, prostaglandin synthesis reduction, anti-inflammatory and antiplatelet effects |

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Set the runtime to **T4 GPU**: `Runtime` > `Change runtime type` > `T4 GPU`
3. Run all cells: `Runtime` > `Run all`
4. The Gradio interface will launch with a shareable link at the end

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/blessinghirwa/Summative.git
cd Summative

# Install dependencies
pip install transformers peft datasets evaluate bitsandbytes trl gradio rouge-score nltk accelerate

# Open the notebook
jupyter notebook medical_assistant.ipynb
```

> **Note**: Local setup requires an NVIDIA GPU with at least 8 GB VRAM and CUDA support.

## Project Structure

```
Summative/
├── medical_assistant.ipynb    # Main notebook (runs end-to-end on Colab)
├── README.md                  # This file
└── Domain-Specific Assistant via LLMs Fine-Tuning.pdf  # Assignment specification
```

## Technical Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU (T4 or better)
- ~8 GB GPU VRAM minimum (4-bit quantization)

### Key Libraries
- `transformers` — Model loading and tokenization
- `peft` — LoRA adapter configuration
- `trl` — Supervised fine-tuning trainer (SFTTrainer)
- `bitsandbytes` — 4-bit quantization
- `datasets` — Dataset loading from Hugging Face
- `evaluate` — ROUGE metric computation
- `nltk` — BLEU score computation
- `gradio` — Interactive chat UI

## Limitations

- **Model size**: TinyLlama (1.1B params) has inherent reasoning limitations compared to larger models
- **Dataset scope**: Medical flashcards cover broad topics but may lack depth in specialized areas
- **No clinical validation**: Responses have not been verified by medical professionals
- **Educational only**: This tool should NOT be used for medical diagnosis or treatment decisions

## Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) by Zhang et al.
- [MedAlpaca](https://github.com/kbressem/medAlpaca) for the medical flashcards dataset
- [Hugging Face](https://huggingface.co/) for the transformers, peft, and trl libraries
