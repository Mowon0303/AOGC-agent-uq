"""Response generators: turn a MIRAGE message list into an agent response.

- ``EchoGenerator``: no model; returns a canned/derived string. For CPU tests.
- ``HFGenerator``: local HF causal LM in 4-bit (bitsandbytes). Colab T4 / 3060.
  CUDA-only for 4-bit; lazy-loads so importing this module never needs torch.
"""

from __future__ import annotations

from typing import Callable


def flatten_messages(messages: list[dict]) -> list[dict]:
    """Normalize MIRAGE messages to [{role, content:str}] for a chat template.

    MIRAGE content is either a str or a list of {type, text} parts (some image
    parts). We join text parts and drop images (text-only rollout).
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p, str):
                    parts.append(p)
            text = "\n".join(parts)
        else:
            text = "" if c is None else str(c)
        out.append({"role": role, "content": text})
    return out


class ResponseGenerator:
    """Interface: messages -> response text."""

    def generate(self, messages: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_n(self, messages: list[dict], n: int) -> list[str]:
        """n samples (for self-consistency / semantic-entropy baselines later)."""
        return [self.generate(messages) for _ in range(n)]


class EchoGenerator(ResponseGenerator):
    """Stub generator for tests/dry-runs. Returns a constant or fn(messages)."""

    def __init__(self, response: str = "<think>ok</think>\n<action>noop()</action>",
                 fn: Callable[[list[dict]], str] | None = None):
        self.response = response
        self.fn = fn

    def generate(self, messages: list[dict]) -> str:
        return self.fn(messages) if self.fn else self.response


class HFGenerator(ResponseGenerator):
    """Local HF causal LM, 4-bit by default (Colab T4 / RTX 3060). Lazy-loaded."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 load_in_4bit: bool = True, device: str | None = None,
                 max_new_tokens: int = 512, temperature: float = 0.0,
                 max_input_tokens: int = 8192):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_cuda = torch.cuda.is_available()
        kwargs = {}
        if self.load_in_4bit:
            if not has_cuda:
                raise RuntimeError(
                    "load_in_4bit needs CUDA (Colab T4 / 3060). On Mac use "
                    "EchoGenerator, or set load_in_4bit=False for a small fp model."
                )
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if has_cuda else torch.float32
            kwargs["device_map"] = "auto" if has_cuda else None

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        if self.device is None:
            self.device = "cuda" if has_cuda else "cpu"

    def generate(self, messages: list[dict]) -> str:
        return self.generate_n(messages, 1)[0]

    def generate_n(self, messages: list[dict], n: int = 1) -> list[str]:
        self._ensure_loaded()
        import torch

        msgs = flatten_messages(messages)
        inputs = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            truncation=True, max_length=self.max_input_tokens,
        )
        inputs = inputs.to(self._model.device)
        do_sample = self.temperature > 0
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=do_sample,
                          num_return_sequences=n,
                          pad_token_id=self._tok.eos_token_id)
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = self._model.generate(inputs, **gen_kwargs)
        prompt_len = inputs.shape[1]
        return [self._tok.decode(o[prompt_len:], skip_special_tokens=True) for o in out]
