from dataset import EVALDATASET
from evaluate import RAGEvaluator
from finetuning.embedding_model.services.model import EmbeddModel
from QA_RAG.main import run_rag_pipeline

eval_dataset_path = "/model/QAs_dataset/dataset.jsonl"
dataset = EVALDATASET()
processed_dataset = dataset.load_eval_dataset(file_path=eval_dataset_path)

print(f"Total QA Pairs: {len(processed_dataset)}")

embedd_ftmodel_ckpt = "/models/finetuning_embeddModel/bge-base-en-v1.5-matryoshka"
embedding_model = EmbeddModel(model_id=embedd_ftmodel_ckpt)
ft_embedd_model = embedding_model.load(eval=True)

evaluator = RAGEvaluator(similarity_model=ft_embedd_model)

for qa_pair in processed_dataset:
    results = []

    question = "what components are typically included in an ai risk management framework?"
    ground_truth = "components such as risk identification, risk assessment, risk mitigation strategies, monitoring and control mechanisms, and incident response protocols are typically included in an ai risk management framework to manage risks throughout the ai lifecycle."

    generated_answer, context = run_rag_pipeline(query=question)

    eval_results = evaluator.evaluate_rag(question=question,
                                ground_truth=ground_truth,
                                rag_response=generated_answer
                                )
    
    results.append(eval_results)
    print(eval_results)
    break