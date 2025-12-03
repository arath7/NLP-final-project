import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

        # CLIP text embeddings are 512-dimensional - matches image encoder
        self.output_dim = 512  

    def forward(self, sentences):
        """
        sentences: list of raw strings, batch size = B
        returns: (B, 512) text embeddings
        """

        tokens = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = {k: v.to(next(self.model.parameters()).device) for k, v in tokens.items()}

        outputs = self.model(**tokens)
        
        # CLIP’s text embedding is the CLS token embedding
        sentence_features = outputs.last_hidden_state[:, 0, :]  # shape (B, 512)

        return sentence_features
