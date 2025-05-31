from finetuning.base_llm.services.llm import LLMService
from finetuning.embedding_model.services.model import EmbeddModel
from services.dataset import DatasetLoader, DataProcessor
from services.embeddings import LocalEmbeddingFunction
from services.vectorstore import VectorStoreManager
from services.chatbot import RAGChatbot
from services.utils import load_ft_model
import os

def run_rag_pipeline(query=None):

    # Initialize models
    model_name = "microsoft/phi-2"
    llm_path = "./base_models/phi-2"
    LLM = LLMService(base_model_id=model_name,
                     local_dir=llm_path)
    base_model, tokenizer, eval_tokenizer = LLM.load_local_llm()
    print("Local Base Model Loaded Successfully!")

    ft_ckpt = "/models/finetuning_QAs/Finetuning Checkpoints 2.0/final-checkpoint/checkpoint-12450"
    ft_model = load_ft_model(base_model=base_model, ft_ckpt=ft_ckpt)

    embedd_ftmodel_ckpt = "/models/finetuning_embeddModel/bge-base-en-v1.5-matryoshka"
    embedding_model = EmbeddModel(model_id=embedd_ftmodel_ckpt)
    ft_embedd_model = embedding_model.load(eval=True)

    # Load and preprocess dataset
    dataset_loader = DatasetLoader()
    data_processor = DataProcessor()

    dataset_path = "/data/AIbooks_dataset/text_finetuningData.jsonl"
    dataset = dataset_loader.load_dataset(dataset_path)
    proc_dataset = data_processor.preprocess_dataset(data=dataset)
    print(f"Length of Dataset: {len(proc_dataset)}")

    # Initialize vector store
    db_path = "/vectorDB/textbooks_v1"
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    embedding_function = LocalEmbeddingFunction(embedd_model=ft_embedd_model)
    vector_store_manager = VectorStoreManager(embedding_function=embedding_function)
    
    vec_DB = vector_store_manager.create_vecdb(
        path=db_path,
        dataset=proc_dataset,
        create_new=False
    )

    # Initialize chatbot
    chatbot = RAGChatbot(model=ft_model, tokenizer=eval_tokenizer)

    # Process query
    if query is None:
        query = "What is the purpose of log anomaly detection?"

    print(f"\nUser:\n{query}")
    response, context = chatbot.process_query(user_query=query, vec_db=vec_DB)
    print(f"\nChatbot:\n{response}")

    return response, context

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)