import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

def evaluate_rag(question, ground_truth, rag_response, similarity_model):
    # Compute BLEU Score
    bleu_score = sentence_bleu([ground_truth.split()], rag_response.split())

    # Compute ROUGE Score
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, rag_response)

    # Compute Embedding Similarity
    emb1 = similarity_model.encode(ground_truth, convert_to_tensor=True)
    emb2 = similarity_model.encode(rag_response, convert_to_tensor=True)
    embedding_similarity = util.pytorch_cos_sim(emb1, emb2).item()
    
    results = {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": rag_response,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "embedding_similarity": embedding_similarity,
    }

    return results