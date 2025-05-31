from base_RAG1.rag_eval.eval_dataset import prepare_rag_evaluation_data
from base_RAG1.rag_main import run_rag_pipeline
from finetuning_embeddModel.embedd_finetuning import load_model
from base_RAG1.rag_eval.evaluator import evaluate_rag

eval_dataset_path = "synthetic_QAs.json"

processed_dataset = prepare_rag_evaluation_data(file_path=eval_dataset_path)

print(f"Total QA Pairs: {len(processed_dataset)}")
# print(processed_dataset[2])

embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka"
embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

for qa_pair in processed_dataset:
    results = []

    question = qa_pair["question"]
    ground_truth = qa_pair["answer"]

    generated_answer, context = run_rag_pipeline(query=question)

    eval_results = evaluate_rag(question=question,
                                ground_truth=ground_truth,
                                rag_response=generated_answer,
                                similarity_model=embedd_model)
    
    results.append(eval_results)
    print(eval_results)
    break