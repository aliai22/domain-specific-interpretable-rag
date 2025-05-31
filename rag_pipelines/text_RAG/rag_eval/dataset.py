import json
import random
import os
from typing import List, Dict

class EVALDATASET:

    def __init__(self) -> None:
        pass

    def load_eval_dataset(self, file_path: str, sample_size: int = 50) -> List[Dict]:
        """
        Prepares a subset of the dataset for RAG evaluation.

        Args:
            file_path (str): Path to the JSONL dataset.
            sample_size (int): Number of QA pairs to sample.

        Returns:
            List[Dict]: Processed dataset with 'question' and 'answer'.
        """
        dataset = self._load_jsonl_dataset(file_path)
        sampled_qa_pairs = self._sample_evaluation_data(dataset, sample_size)

        processed_data = []
        for qa_pair in sampled_qa_pairs:
            processed_data.append({
                "question": self._preprocess_text(qa_pair["question"]),
                "answer": self._preprocess_text(qa_pair["answer"])
            })
        
        return processed_data

    def _load_jsonl_dataset(self, file_path: str) -> List[Dict]:
        """
        Loads a JSONL file containing question-answer pairs.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            List[Dict]: A list of dictionaries with 'question' and 'answer' fields.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        dataset = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        
        return dataset

    def _preprocess_text(self, text: str) -> str:
        """
        Cleans and normalizes text by converting to lowercase and stripping whitespace.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        return " ".join(text.lower().strip().split())

    def _sample_evaluation_data(self, dataset: List[Dict], sample_size: int = 50) -> List[Dict]:
        """
        Samples a subset of QA pairs randomly and shuffles them.

        Args:
            dataset (List[Dict]): The full dataset of QA pairs.
            sample_size (int): Number of samples to extract.

        Returns:
            List[Dict]: A shuffled list of sampled QA pairs.
        """
        sampled_data = random.sample(dataset, min(sample_size, len(dataset)))
        random.shuffle(sampled_data)  # Ensure different QA types are mixed
        return sampled_data