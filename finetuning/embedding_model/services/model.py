import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

class EmbeddModel:
    
    def __init__(self, model_id) -> None:
        self.model_id = model_id

    def load(self, eval=False):
        if not eval:
            model = SentenceTransformer(
                self.model_id,
                # device="cuda" if torch.cuda.is_available() else "cpu",
                model_kwargs={"attn_implementation": "sdpa"},
                model_card_data=SentenceTransformerModelCardData(
                    language="en",
                    license="apache-2.0",
                    model_name="BGE base Financial Matryoshka",
                ),
            )
        else:
            model = SentenceTransformer(
                self.model_id,
                device = "cuda" if torch.cuda.is_available() else "cpu"
            )
        # model.tokenizer.pad_token = model.tokenizer.eos_token  # Assign EOS token as PAD token
        print("Model Loaded Successfully!")
        return model