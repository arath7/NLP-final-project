from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch

class SBERTEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.proj = nn.Linear(768, 512)  # SBERT output dim → 512 to match image encoder

    def forward(self, sentences):
        """
        sentences: list of raw text strings, len = batch size
        """
        embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            device=self.proj.weight.device,
            show_progress_bar=False
        )
        return self.proj(embeddings)    # (B, 512)
