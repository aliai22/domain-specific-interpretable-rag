from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util

class RAGEvaluator:
    def __init__(self, similarity_model):
        self.similarity_model = similarity_model
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def _compute_bleu(self, reference: str, hypothesis: str) -> float:
        return sentence_bleu([reference.split()], hypothesis.split())

    def _compute_rouge(self, reference: str, hypothesis: str) -> dict:
        rouge_scores = self.rouge.score(reference, hypothesis)
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        }

    def _compute_embedding_similarity(self, reference: str, hypothesis: str) -> float:
        emb1 = self.similarity_model.encode(reference, convert_to_tensor=True)
        emb2 = self.similarity_model.encode(hypothesis, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def evaluate_rag(self, question: str, ground_truth: str, rag_response: str) -> dict:
        bleu = self._compute_bleu(ground_truth, rag_response)
        rouge = self._compute_rouge(ground_truth, rag_response)
        similarity = self._compute_embedding_similarity(ground_truth, rag_response)

        return {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": rag_response,
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"],
            "embedding_similarity": similarity
        }