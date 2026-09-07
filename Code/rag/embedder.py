"""
Turning text into vectors.

One encoder is used for both queries and passages. That is the dual-encoder
arrangement of dense passage retrieval: query and passage land in the same space,
so relevance is a dot product.

Mean pooling over the last hidden layer, then L2 normalisation, so that an inner
product equals cosine similarity and the index can use the cheaper operation.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import config


class Embedder:
    """Sentence encoder. Loaded once, reused for queries and passages.

    Parameters
    ----------
    model_id : str, optional
        Defaults to :data:`config.EMBEDDING_MODEL`. Changing it between runs
        invalidates comparison, because retrieval would no longer be fixed.
    device : str, optional
        ``"cuda"``, ``"mps"``, ``"cpu"``. Autodetected when omitted.
    """

    def __init__(self, model_id=None, device=None, max_len=None):
        self.model_id = model_id or config.EMBEDDING_MODEL
        self.max_len = max_len or config.MAX_SEQ_LEN
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    @property
    def dim(self):
        return self.model.config.hidden_size

    def encode(self, texts, batch_size=32):
        """Encode a list of strings into an ``(n, dim)`` float32 array.

        Vectors are L2-normalised, so ``a @ b.T`` is cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                out.append(pooled.cpu().numpy())
        return np.vstack(out).astype("float32")
