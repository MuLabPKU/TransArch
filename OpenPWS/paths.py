"""Local Hugging Face checkpoints and prompt-generation resources."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get('DEEPSEEK_MODELS', ROOT/'models'))
MODEL_V4 = Path(os.environ.get('DEEPSEEK_V4_MODEL', MODELS/'DeepSeek-V4-Flash-0731'))
MODEL_V41 = Path(os.environ.get('DEEPSEEK_V41_MODEL', MODELS/'DeepSeek-V4.1-Flash'))
FILLER_SOURCE = ROOT/'assets/filler-train.txt.gz'
