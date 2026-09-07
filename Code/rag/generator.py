"""
The generation stage.

Wraps a causal language model behind one call. The model, its decoding
temperature and its output length are fixed in :mod:`config` and must not vary
between conditions: the experiment changes the prompt and nothing else.

Only the generator sees the system prompt. The retriever never does.
"""

import config


class Generator:
    """Local instruct model.

    Parameters
    ----------
    model_id : str, optional
        Defaults to :data:`config.GENERATOR_MODEL`.
    temperature, max_new_tokens : float, int, optional
        Default to the fixed values in :mod:`config`. Override only for a smoke
        test, never between experimental conditions.
    """

    def __init__(self, model_id=None, temperature=None, max_new_tokens=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.model_id = model_id or config.GENERATOR_MODEL
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto").eval()

    def __call__(self, prompt):
        """Generate one answer. Returns ``(text, n_new_tokens)``.

        Each call is independent. No state carries between calls, so the order
        in which cells are run cannot influence the results.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), len(new)
