"""
The generation stage, local backend.

Wraps a Hugging Face causal language model behind one call, with the same
interface as :class:`rag.openai_generator.OpenAIGenerator`. The experiment
uses the API backend (:data:`config.GENERATOR_BACKEND`); this one is kept for
smoke tests and for running without network access, and a run made with it is
not comparable with a run made with the API. The decoding temperature and the
output length are fixed in :mod:`config` and must not vary between conditions.

Only the generator sees the system prompt. The retriever never does.
"""

import config


class _CountingSanitiser:
    """Records how often a non-finite logit row reached the sampler.

    Runs after InfNanRemoveLogitsProcessor has already repaired the scores, so
    it counts occurrences without changing anything. The count is written with
    each generation: a run where this is always zero needed no intervention,
    and one where it is not lets a reader see exactly which outputs were
    affected instead of having to trust that it did not matter.
    """

    def __init__(self):
        self.n_nonfinite = 0

    def __call__(self, input_ids, scores):
        import torch
        if not torch.isfinite(scores).all():
            self.n_nonfinite += 1
        return scores


def default_dtype(torch, device):
    """The dtype to load the generator in, from :data:`config.GENERATOR_DTYPE`.

    Kept in config rather than decided here, because it changes the arithmetic
    the model does and is therefore a controlled parameter: it must be the same
    for every condition, and it is recorded with every generation.

    float32 would remove the MPS non-finite-logit problem outright, but 7B
    parameters is ~28 GB in float32 against ~15 GB in bfloat16, so it is not
    available at the size this experiment runs at. The logits processor in
    :class:`Generator` handles the problem instead, at reduced precision, and
    counts how often it fired.
    """
    name = getattr(config, "GENERATOR_DTYPE", "auto")
    if name in (None, "auto"):
        return "auto"
    return getattr(torch, name) if isinstance(name, str) else name


def pick_device(torch):
    """The best device available, preferring an accelerator over the CPU.

    Which device is used does not change what the experiment measures -- the
    model and its parameters are the same -- but it changes how long 30,000
    generations take by a large factor, so it is chosen rather than left to
    whatever the default happens to be.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Generator:
    """Local instruct model.

    Parameters
    ----------
    model_id : str, optional
        Defaults to :data:`config.LOCAL_GENERATOR_MODEL`.
    temperature, max_new_tokens : float, int, optional
        Default to the fixed values in :mod:`config`. Override only for a smoke
        test, never between experimental conditions.
    """

    def __init__(self, model_id=None, temperature=None, max_new_tokens=None,
                 device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.model_id = model_id or config.LOCAL_GENERATOR_MODEL
        self.backend = "local"
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # device_map="auto" needs accelerate, which is not a dependency of the
        # experiment: nothing here shards a model across devices. When it is
        # absent the model is loaded plainly and moved once, which is the same
        # thing for a single-device run. Falling back rather than requiring the
        # package keeps a 30,000-generation run from dying at import time over
        # an optional dependency.
        self.device = device or pick_device(torch)
        self.dtype = dtype or default_dtype(torch, self.device)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype, device_map="auto").eval()
        except (ImportError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype).to(self.device).eval()

        # Sampling draws from softmax(logits). In reduced precision on MPS the
        # logits can come back with a non-finite entry on a long context, and
        # torch.multinomial then raises "probability tensor contains either
        # inf, nan or element < 0" and kills the run. This processor replaces
        # those entries before the draw. It fires rarely; when it does, the
        # generation is recorded as sanitised rather than silently kept, since
        # a numerical intervention that changed an output must be visible in
        # the log rather than buried in the library.
        from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
        self._sanitiser = _CountingSanitiser()
        self._processors = LogitsProcessorList(
            [InfNanRemoveLogitsProcessor(), self._sanitiser])

    def __call__(self, prompt):
        """Generate one answer. Returns ``(text, n_new_tokens)``.

        Each call is independent. No state carries between calls, so the order
        in which cells are run cannot influence the results.

        Parameters
        ----------
        prompt : rag.prompts.Prompt or str
            A :class:`~rag.prompts.Prompt` is rendered with its system turn in
            the *system* role, which is what makes S a standing instruction
            rather than part of the user's message. A bare string is treated as
            a user turn with no system prompt, which is only for smoke tests:
            the experiment always passes a Prompt.
        """
        if hasattr(prompt, "as_messages"):
            messages = prompt.as_messages()
        else:
            messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        before = self._sanitiser.n_nonfinite
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=self._processors,
            )
        new = out[0][enc["input_ids"].shape[1]:]
        self.last_sanitised = self._sanitiser.n_nonfinite - before
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), len(new)

    def generate(self, prompt):
        """Same shape as the API backend's :meth:`generate`.

        The local backend is single-threaded (a smoke-test path), so the
        sanitiser counter is read off the instance here; the API backend
        returns its provenance instead, because it is shared across threads.
        """
        text, n_tokens = self(prompt)
        return {"text": text, "n_tokens": n_tokens,
                "system_fingerprint": None, "finish_reason": None,
                "n_sanitised_steps": self.last_sanitised}
