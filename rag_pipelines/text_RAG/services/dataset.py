import jsonlines
from typing import List, Generator

class CustomString(str):
    """Custom string class that includes metadata."""
    def __new__(cls, content, metadata=None):
        obj = super().__new__(cls, content)
        obj.page_content = content
        obj.metadata = metadata
        return obj

class DatasetLoader:
    def __init__(self):
        pass

    def load_dataset(self, path: str) -> List[str]:
        """
        Load dataset from a jsonlines file.
        
        Args:
            path (str): Path to the jsonlines file
            
        Returns:
            List[str]: List of QA pairs in text format
        """
        qa_pairs = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                qa_text = f"Q: {obj['question']}\nA: {obj['answer']}"
                qa_pairs.append(qa_text)
        print("Dataset Loaded Successfully!")
        return qa_pairs

class DataProcessor:
    def __init__(self):
        pass

    def preprocess_dataset(self, data: List[str]) -> List[CustomString]:
        """
        Preprocess the dataset by converting each item to a CustomString with metadata.
        
        Args:
            data (List[str]): List of text data
            
        Returns:
            List[CustomString]: List of CustomString objects with metadata
        """
        qa_chunks = [CustomString(content, {"index": idx}) for idx, content in enumerate(data)]
        print("Dataset Preprocessed Successfully!")
        return qa_chunks

    def batchify(self, data: List, batch_size: int) -> Generator:
        """
        Yield successive batches from the dataset.
        
        Args:
            data (List): List of data to batch
            batch_size (int): Size of each batch
            
        Yields:
            Generator: Batches of data
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]