import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
device_map = {"":0}

class LLMService:

    def __init__(self, base_model_id: str, local_dir: str):
        self.model_id = base_model_id
        self.model_path_local = local_dir

    # Load Pretrained Model
    def load_llm(self):
        original_model = AutoModelForCausalLM.from_pretrained(self.model_id, 
                                                            device_map=device_map,
                                                            quantization_config=bnb_config,
                                                            trust_remote_code=True,
                                                            )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                trust_remote_code=True,
                                                padding_side='left',
                                                add_eos_toke=True,
                                                add_bos_token=True,
                                                use_fast=False
                                                )
        tokenizer.pad_token = tokenizer.eos_token

        eval_tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                    add_bos_token=True,
                                                    trust_remote_code=True,
                                                    use_fast=False
                                                    )
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        
        return original_model, tokenizer, eval_tokenizer

    # Save Model Locally
    def save_llm(self,
                 model,
                 tokenizer,
                 eval_tokenizer
                 ):
        model.save_pretrained(f"{self.model_path_local}/llm/")
        tokenizer.save_pretrained(f"{self.model_path_local}/tokenizer/")
        eval_tokenizer.save_pretrained(f"{self.model_path_local}/evaltokenizer/")

    # Load Local LLM
    def load_local_llm(self,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(f"{self.model_path_local}/llm/")
        model.to(device)

        # Load the tokenizer
        # if self.model_id:
        #     tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        #     tokenizer.pad_token = tokenizer.eos_token
        #     eval_tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        #     eval_tokenizer.pad_token = eval_tokenizer.eos_token
        # else:
        
        tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path_local}/tokenizer/", trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        eval_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path_local}/evaltokenizer/", trust_remote_code=True, use_fast=False)
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

        return model, tokenizer, eval_tokenizer