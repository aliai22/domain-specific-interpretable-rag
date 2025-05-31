from finetuning.base_llm.services.llm import LLMService
from finetuning.embedding_model.services.model import EmbeddModel
from services.dataset import DatasetLoader, DataProcessor
from services.embeddings import LocalEmbeddingFunction
from services.vectorstore import VectorStoreManager
from services.chatbot import RAGChatbot
from services.utils import load_ft_model
from services.keyword_service import KeywordExtractor, KeywordDatabase, DocumentReranker
import os

def format_context_to_prompt(documents, top_n=3):
    """
    Format the reranked documents into a prompt string.
    
    Args:
        documents: List of documents to format
        top_n: Number of top documents to include
        
    Returns:
        str: Formatted prompt string
    """
    prompt = ""
    for i, doc in enumerate(documents[:top_n]):
        prompt += f"{i+1}. {doc.page_content.strip()}\n"
    return prompt

def run_rag_pipeline(query=None, use_keyword_reranking=False, keyword_db_path=None, top_n_docs=5):
    """
    Run the RAG pipeline with optional keyword-based document reranking.
    
    Args:
        query (str, optional): User query. Defaults to None.
        use_keyword_reranking (bool, optional): Whether to use keyword-based reranking. Defaults to False.
        keyword_db_path (str, optional): Path to the keyword database JSON file. Required if use_keyword_reranking is True.
        top_n_docs (int, optional): Number of top documents to use for response generation. Defaults to 3.
    """
    # Initialize models
    model_name = "microsoft/phi-2"
    llm_path = "./base_models/phi-2"
    LLM = LLMService(base_model_id=model_name,
                     local_dir=llm_path)
    base_model, tokenizer, eval_tokenizer = LLM.load_local_llm()
    print("Local Base Model Loaded Successfully!")

    ft_ckpt = "/models/finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-5000"
    ft_model = load_ft_model(base_model=base_model, ft_ckpt=ft_ckpt)

    embedd_ftmodel_ckpt = "/models/finetuning_embeddModel/bge-base-en-v1.5-matryoshka"
    embedding_model = EmbeddModel(model_id=embedd_ftmodel_ckpt)
    ft_embedd_model = embedding_model.load(eval=True)

    # Load and preprocess dataset
    dataset_loader = DatasetLoader()
    data_processor = DataProcessor()
    
    dataset_path = "/data/QAs_dataset/dataset.jsonl"
    dataset = dataset_loader.load_dataset(dataset_path)
    proc_dataset = data_processor.preprocess_dataset(data=dataset)
    print(f"Length of Dataset: {len(proc_dataset)}")

    # Initialize vector store
    db_path = "/vectorDB/QAs_v1"
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

    _, context = chatbot.process_query(user_query=query, vec_db=vec_DB)

    # Apply keyword-based reranking if enabled
    if use_keyword_reranking:
        if keyword_db_path is None:
            raise ValueError("keyword_db_path must be provided when use_keyword_reranking is True")
            
        # Initialize keyword-based reranking components
        keyword_extractor = KeywordExtractor()
        keyword_db = KeywordDatabase(keyword_db_path)
        document_reranker = DocumentReranker()

        # Extract keywords from query
        query_keywords = keyword_extractor.extract_keywords(query)
        print(f"\nExtracted Keywords: {list(query_keywords.keys())}")

        # Rerank documents
        reranked_context = document_reranker.rerank_documents(
            retrieved_docs=context,
            query_keywords=list(query_keywords.keys()),
            keyword_db=keyword_db
        )

        # Format reranked documents into prompt
        formatted_context = format_context_to_prompt(reranked_context, top_n=top_n_docs)
        
        # Generate response using the formatted context
        response, context = chatbot.process_query(user_query=query, vec_db=vec_DB, context=formatted_context)

    # Get response and context from retrieved documents
    response, context = chatbot.process_query(user_query=query, vec_db=vec_DB)

    print(f"\nChatbot:\n{response}")

    return response, context

if __name__ == "__main__":
    
    # Example usage with keyword reranking
    # response, context = run_rag_pipeline(
    #     query="What is uncertainty estimation?",
    #     use_keyword_reranking=True,
    #     keyword_db_path="/path/to/keywords_database.json",
    #     top_n_docs=5
    # )
    
    # Example usage without keyword reranking
    response, context = run_rag_pipeline(query=None)