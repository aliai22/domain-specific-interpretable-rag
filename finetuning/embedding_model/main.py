from services.dataset import DatasetService
from services.utils import generate_evaluator, define_loss, prepare_trainer
from services.model import EmbeddModel
from base_llm.services.gpu_monitor import GPUMonitor
from base_llm.services.utils import clear_cache

dataset_path = "./data/dataset.jsonl"
output_path = "./data/EmbedModel_Finetuning/filtered_qa_pairs.jsonl"

DATASET = DatasetService()

DATASET.preprocess_qas(input_file=dataset_path,
                       output_file=output_path)

train_dataset, test_dataset, corpus, queries, relevant_docs = DATASET.load_dataset_with_structure(test_path="/data/EmbedModel_Finetuning/test_dataset.json",
                                                                                                  train_path="/data/EmbedModel_Finetuning/train_dataset.json")

evaluator = generate_evaluator(queries=queries,
                               corpus=corpus,
                               relevant_docs=relevant_docs)

embedd_model_id = "BAAI/bge-base-en-v1.5"
embedding_model = EmbeddModel(model_id=embedd_model_id)

base_model = embedding_model.load()

trainer_loss = define_loss(model=base_model)

output_dir = "/models/finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"

trainer = prepare_trainer(model=base_model,
                train_dataset=train_dataset,
                train_loss=trainer_loss,
                evaluator=evaluator,
                output_dir=output_dir)

clear_cache()

# Log file
gpu_log_file = "./gpu_logs/finetuning/embedding_model/gpu_training_log.csv"

GPU_logs = GPUMonitor(log_file=gpu_log_file,
                         )

# Start GPU monitoring
GPU_logs.start_monitoring()

# Start training
print("Training Started!")
trainer.train() # use resume_from_checkpoint="./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0/checkpoint-5000" to resume finetuning from a checkpoint

# Stop GPU monitoring
GPU_logs.stop_monitoring()

# save the best model
trainer.save_model()