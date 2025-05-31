from bert_score import BERTScorer
import torch
from peft import PeftModel
from transformers import set_seed
import random

from services.llm import LLMService
from services.dataset import DatasetService
from services.utils import get_max_len, gen_response

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_name = "microsoft/phi-2"
llm_path = "./base_models/phi-2"

LLM = LLMService(base_model_id=model_name,
                 local_dir=llm_path)

base_model, tokenizer, eval_tokenizer = LLM.load_local_llm()

print("Base Model Loaded Successfully from Local Directory!")

dataset_path = "./data/dataset.jsonl"

DATASET = DatasetService()

dataset = DATASET.load(file_path=dataset_path)
print("Data File Loaded Successfully!")

splitted_dataset = DATASET.split(dataset=dataset)
print("Data Split Created Successfully!")

logger.info(f"Training Dataset: {splitted_dataset['train']}")
logger.info(f"Training Dataset: {splitted_dataset['test']}")

max_length = get_max_len(model=base_model)

seed = 42
set_seed(seed)

print("Dataset Preprocessing .....")

train_dataset = DATASET.preprocess(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["train"])
eval_dataset = DATASET.preprocess(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["test"])

print("Dataset Preprocessed!")

finetuned_model_path = ".models/Finetuning_Checkpoints_3.0/final-checkpoint/checkpoint-5000"
ft_model = PeftModel.from_pretrained(base_model,
                                     finetuned_model_path,
                                     torch_dtype=torch.float16,
                                     is_trainable=False)

i = random.randint(0, len(splitted_dataset['test']))
print(f"Sample Index: {i}")

question = splitted_dataset['test'][15679]['question']
ground_truth_response = splitted_dataset['test'][15679]['answer']

prompt = f"Instruct: Answer the following question accurately.\n{question}\nOutput:\n"

peft_model_res = gen_response(model=ft_model,
                     prompt=prompt,
                     tokenizer=eval_tokenizer,
                     )
peft_model_output = peft_model_res[0].split('Output:\n')[1]

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE Answer:\n{ground_truth_response}\n')
print(dash_line)
print(f'PEFT MODEL Response:\n{peft_model_output}')

scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score([ground_truth_response], [peft_model_output])
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")