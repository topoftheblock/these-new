"""
The generation stage, through the OpenAI chat completions API.

Same interface as :class:`rag.generator.Generator`: call it with a
:class:`rag.prompts.Prompt` and get ``(text, n_new_tokens)`` back. The system
turn goes in the ``system`` role and the user turn in the ``user`` role, which is
what makes S a standing instruction rather than part of the user's message.

The model, its temperature and its output length are fixed in :mod:`config` and
must not vary between conditions. Each call is an independent request: nothing
is cached or reused between cells, and no seed is sent, so the repeats are
independent draws from the same distribution.

Why this does not use the ``openai`` package
--------------------------------------------

On the machine this was written for, ``import openai`` fails because the
environment carries a pydantic / pydantic-core version conflict (the same one
that stops ``datasets`` from importing, see ``preprocessing/huggingface.py``).
The chat completions endpoint is one HTTPS POST, so the standard library is
enough and no client dependency is needed.

The key
-------

Read from the environment variable named in :data:`config.OPENAI_ENV_VAR`, and
failing that from a ``.env`` file in the ``Code/`` directory. It is never
written into source and never logged.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_api_key(env_var=None, env_file=_ENV_FILE):
    """Return the API key, or raise ``SystemExit`` with instructions.

    Order: the environment variable, then ``Code/.env``. The ``.env`` file is
    parsed with python-dotenv when it is installed and by hand otherwise, so
    the dependency stays optional.
    """
    env_var = env_var or config.OPENAI_ENV_VAR
    key = os.environ.get(env_var)
    if not key and env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, value = line.split("=", 1)
                    os.environ.setdefault(name.strip(), value.strip().strip("'\""))
        key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"no API key. Set {env_var} in the environment or put\n"
            f"    {env_var}=sk-...\n"
            f"in {env_file} (see .env.example). The key is never stored in source.")
    return key


class OpenAIGenerator:
    """Chat completions behind one call.

    Parameters
    ----------
    model_id : str, optional
        Defaults to :data:`config.GENERATOR_MODEL`.
    temperature, max_new_tokens : float, int, optional
        Default to the fixed values in :mod:`config`. Override only for a smoke
        test or for the judge, never between experimental conditions.
    api_key : str, optional
        Defaults to :func:`load_api_key`.

    Thread safety
    -------------
    One instance is shared by every worker thread of a concurrent run. The
    per-request metadata is therefore *returned* by :meth:`generate` rather
    than stashed on the instance: an attribute like ``last_fingerprint`` would
    be overwritten by another thread between a call and the read of it, and the
    provenance recorded against a generation would silently belong to a
    different request. The only mutable state left is two counters, which are
    incremented under a lock.
    """

    def __init__(self, model_id=None, temperature=None, max_new_tokens=None,
                 api_key=None, base_url=None, timeout=None, max_retries=None):
        self.model_id = model_id or config.GENERATOR_MODEL
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        self.api_key = api_key or load_api_key()
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.timeout = timeout or config.OPENAI_TIMEOUT_S
        self.max_retries = (config.OPENAI_MAX_RETRIES if max_retries is None
                            else max_retries)
        self.dtype = "api"
        self.backend = "openai"
        self.last_sanitised = 0        # the local generator's counter; never fires here
        self._lock = threading.Lock()  # guards the two counters only
        self.n_calls = 0
        self.n_retries = 0

    # --------------------------------------------------------------- HTTP
    def _post(self, payload):
        """POST to /chat/completions with retries on rate limits and outages.

        Retries on 408, 409, 429 and 5xx and on network errors, with
        exponential backoff and the server's ``Retry-After`` when it sends one.
        Any other status is a caller error and is raised at once.
        """
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers,
                                             method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                status = exc.code
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                retryable = status in (408, 409, 429) or status >= 500
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenAI API returned {status}: {detail}") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after else delay
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI API unreachable: {exc}") from None
                wait = delay
            with self._lock:
                self.n_retries += 1
            time.sleep(min(wait, 60.0))
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")      # the loop always returns or raises

    # --------------------------------------------------------------- call
    def generate(self, prompt):
        """Generate one answer. Returns a dict, safe to call from many threads.

        Keys: ``text``, ``n_tokens``, ``system_fingerprint``, ``finish_reason``.
        The provenance travels with the result rather than being read back off
        the instance, so a concurrent run records the fingerprint of the request
        that actually produced the answer.

        Parameters
        ----------
        prompt : rag.prompts.Prompt or str
            A :class:`~rag.prompts.Prompt` is rendered with its system turn in
            the *system* role. A bare string is sent as a user turn with no
            system prompt, which is what the judge uses: its instruction is
            the whole message and there is no standing prompt to separate.
        """
        if hasattr(prompt, "as_messages"):
            messages = prompt.as_messages()
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_new_tokens,
            "n": 1,
        }
        data = self._post(payload)
        with self._lock:
            self.n_calls += 1

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return {
            "text": (choice.get("message", {}).get("content") or "").strip(),
            "n_tokens": int(usage.get("completion_tokens", 0)),
            "system_fingerprint": data.get("system_fingerprint"),
            "finish_reason": choice.get("finish_reason"),
        }

    def __call__(self, prompt):
        """``(text, n_tokens)``, for callers that want only the answer."""
        out = self.generate(prompt)
        return out["text"], out["n_tokens"]
