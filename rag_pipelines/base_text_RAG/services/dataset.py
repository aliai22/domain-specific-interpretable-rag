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

    def load_dataset(self, path: str):
        all_contents = []
        all_metas = []
        # i=0
        with jsonlines.open(path) as reader:
            for obj in reader:
                if len(obj["content"]) >= 512:
                    all_contents.append(obj["content"])
                    all_metas.append(obj["metadata"])
                # print(len(obj['content']))
                # i+=1
                # if i==10:
                #     break
        print(f"Dataset Loaded Successfully! Total Size {len(all_contents)}")
        return all_contents, all_metas

class DataProcessor:
    def __init__(self):
        pass

    def preprocess_dataset(self, data: List[str], metadata: List) -> List[CustomString]:
        if len(data) != len(metadata):
            raise ValueError("Data and metadata lists must have the same length")

        qa_chunks = [CustomString(content, metadata) for content, metadata in zip(data, metadata)]
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