from peft import PeftModel
import torch

def load_ft_model(base_model, ft_ckpt:str):
    ft_model = PeftModel.from_pretrained(base_model,
                                         ft_ckpt,
                                         torch_dtype=torch.float16,
                                         is_trainable=False)
    print("Finetuned Model Loaded Successfully!")
    return ft_model