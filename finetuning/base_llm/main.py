from services.llm import LLMService
from services.dataset import DatasetService
from services.finetune import FinetuningService
from services.gpu_monitor import GPUMonitor
from services.utils import get_max_len, print_number_of_trainable_model_parameters, clear_cache

from transformers import set_seed
import torch
import os

import time
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_name = "microsoft/phi-2"
llm_path = "./base_models/phi-2"

LLM = LLMService(base_model_id=model_name,
                 local_dir=llm_path)

original_llm, tokenizer, eval_tokenizer = LLM.load_llm()
logger.info("Model Loaded Successfully from HuggingFace!")

if not os.path.exists(f"{llm_path}/llm"):
    LLM.save_llm(original_llm, tokenizer, eval_tokenizer)

logger.info("Model Saved Successfully to Local Directory!")

model, tokenizer, eval_tokenizer = LLM.load_local_llm()

logger.info("Model Loaded Successfully from Local Directory!")

# ------------------------------------------------------------------------------------------------------

dataset_path = "./data/uniqueQA_dataset0.7.jsonl"

DATASET = DatasetService()

dataset = DATASET.load(file_path=dataset_path)
logger.info("Data File Loaded Successfully!")

splitted_dataset = DATASET.split(dataset=dataset)
logger.info("Data Split Created Successfully!")

logger.info(f"Training Dataset: {splitted_dataset['train']}")
logger.info(f"Training Dataset: {splitted_dataset['test']}")

# ------------------------------------------------------------------------------------------------------

max_length = get_max_len(model=model)

seed = 42
set_seed(seed)

logger.info("Dataset Preprocessing .....")

train_dataset = DATASET.preprocess(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["train"])
eval_dataset = DATASET.preprocess(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["test"])

logger.info("Dataset Preprocessed!")

print("Original LLM Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(model))

# ------------------------------------------------------------------------------------------------------

output_dir = "./models//Finetuning_Checkpoints_filtered1.0/final-checkpoint"

FINETUNE = FinetuningService(model=model,
                             output_dir=output_dir,
                             training_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             tokenizer=tokenizer)

peft_model = FINETUNE.peft_model()

print("Peft Model Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(peft_model))

peft_trainer = FINETUNE.peft_trainer(peft_model=peft_model)

# -----------------------------------------------------------------------------------------------------

# ckpt_dir = "Finetuning Checkpoints/final-checkpoint/checkpoint-2000"

clear_cache()

# Log file
gpu_log_file = "./gpu_logs/finetuning/base_llm/gpu_training_log_filtered4.0.csv"

GPU_logs = GPUMonitor(log_file=gpu_log_file,
                         )

# Start GPU monitoring
GPU_logs.start_monitoring()

# Start training
logger.info("Training Started!")
peft_trainer.train()

# Stop GPU monitoring
GPU_logs.stop_monitoring()