from services.utils import generate_evaluator
from services.dataset import DatasetService
from services.model import EmbeddModel

DATASET = DatasetService()

train_dataset, test_dataset, queries, corpus, relevant_docs = DATASET.load_dataset_with_structure(test_path="/data/EmbedModel_Finetuning/test_dataset.json",
                train_path="/data/EmbedModel_Finetuning/train_dataset.json",
                )

evaluator = generate_evaluator(queries=queries,
                   corpus=corpus,
                   relevant_docs=relevant_docs,
                   )

model_id = "BAAI/bge-base-en-v1.5"
base_embedding_model = EmbeddModel(model_id=model_id)

base_model = base_embedding_model.load(eval=True)

baseline_results = evaluator(base_model)
print("Baseline Model")
print("dim_512_cosine_ndcg@10: ", baseline_results["dim_512_cosine_ndcg@10"])

ft_ckpt = "/models/finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
FT_embedding_model = EmbeddModel(model_id=ft_ckpt)

ft_model = FT_embedding_model.load(eval=True)

ft_results = evaluator(ft_model)
print("Finetuned Model")
print("dim_512_cosine_ndcg@10: ", ft_results["dim_512_cosine_ndcg@10"])