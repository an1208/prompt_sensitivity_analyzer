"""Text-generation backends. Every backend maps prompt -> output string."""
from __future__ import annotations

import importlib
import importlib.util
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional

import requests


class Backend(ABC):
    name = "backend"

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 128) -> str:
        ...


# ------------------------------------------------------------------ mock
_STOP = set(
    "a an the of to in on for and or is are was were be do does did can could would should i me my you your we it its "
    "this that what which who how why please tell give explain describe any some with as at by from into about "
    "name list calculate provide".split()
)


class MockBackend(Backend):
    """Deterministic fake 'model' for demos and tests (NOT a language model).

    Output = the sorted content words of the prompt, so paraphrases that share more vocabulary
    yield more similar outputs. With temperature > 0 a seeded random word is dropped.
    """

    name = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 128) -> str:
        words = [w for w in re.findall(r"[a-z0-9']+", prompt.lower()) if w not in _STOP]
        words = sorted(set(words))
        if temperature > 0 and len(words) > 2:
            rng = random.Random(f"{seed}|{prompt}")
            words.pop(rng.randrange(len(words)))
        return "Answer: " + " ".join(words)


# ------------------------------------------------------------------ callable (plug in your own system, e.g. a RAG pipeline)
class CallableBackend(Backend):
    """Wrap any Python function `fn(prompt: str) -> str`, e.g. your RAG assistant's `answer()`.

    spec: "package.module:function"  or  "path/to/file.py:function"
    """

    def __init__(self, spec: str):
        target, _, func = spec.rpartition(":")
        if not target or not func.isidentifier():
            raise ValueError("callable spec must look like 'module:function' or 'file.py:function'")
        if target.endswith(".py") or os.sep in target or "/" in target:
            path = os.path.abspath(target)
            mod_spec = importlib.util.spec_from_file_location("psa_user_module", path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(mod_spec)
            sys.path.insert(0, os.path.dirname(path))
            mod_spec.loader.exec_module(module)
        else:
            sys.path.insert(0, os.getcwd())
            module = importlib.import_module(target)
        fn = getattr(module, func)
        if not callable(fn):
            raise TypeError(f"{spec} is not callable")
        self.fn: Callable[[str], str] = fn
        self.name = f"callable:{spec}"

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 128) -> str:
        return str(self.fn(prompt))


# ------------------------------------------------------------------ Ollama (CPU-friendly local models)
class OllamaBackend(Backend):
    def __init__(self, model: str, host: Optional[str] = None, timeout: float = 600):
        host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.replace("://0.0.0.0", "://localhost").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 128) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_new_tokens},
        }
        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}. Is it running?") from e
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama error: {r.text}")
        return r.json().get("response", "").strip()


# ------------------------------------------------------------------ HuggingFace transformers
class HFBackend(Backend):
    """Local HuggingFace model. Handles encoder-decoder (Flan-T5) and decoder-only (Mistral-7B-Instruct)."""

    def __init__(self, model: str, device: Optional[str] = None, load_in_4bit: bool = False, use_chat_template: bool = True):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("HF backend needs: pip install torch transformers accelerate") from e
        self.torch = torch
        self.name = f"hf:{model}"
        cfg = AutoConfig.from_pretrained(model)
        self.seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
        self.tok = AutoTokenizer.from_pretrained(model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        cls = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        self.model = cls.from_pretrained(model, **kwargs)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.use_chat_template = use_chat_template and (not self.seq2seq) and bool(getattr(self.tok, "chat_template", None))
        if not self.seq2seq and self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int = 0, max_new_tokens: int = 128) -> str:
        torch = self.torch
        torch.manual_seed(seed)
        if self.use_chat_template:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        else:
            enc = self.tok(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen["temperature"] = temperature
        if not self.seq2seq:
            gen["pad_token_id"] = self.tok.pad_token_id
        with torch.no_grad():
            out = self.model.generate(**enc, **gen)
        if not self.seq2seq:
            out = out[:, enc["input_ids"].shape[1]:]
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


def make_backend(kind: str, model: Optional[str] = None, host: Optional[str] = None, **kw) -> Backend:
    if kind == "mock":
        return MockBackend()
    if not model:
        raise ValueError(f"--model is required for backend '{kind}'")
    if kind == "callable":
        return CallableBackend(model)
    if kind == "ollama":
        return OllamaBackend(model, host)
    if kind == "hf":
        return HFBackend(model, **kw)
    raise ValueError(f"unknown backend {kind!r} (mock | callable | ollama | hf)")
