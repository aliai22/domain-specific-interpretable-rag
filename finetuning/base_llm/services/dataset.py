from datasets import load_dataset
from functools import partial
from services.utils import create_prompt_formats, preprocess_batch

class DatasetService:

    def __init__(self) -> None:
        pass

    def load(self,
             file_path: str):
        dataset = load_dataset("json",
                               data_files=file_path)
        return dataset

    def split(self,
              dataset):
        split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        return split_dataset

    def preprocess(self,
                    tokenizer,
                    max_length: int,
                    seed,
                    dataset):
        
        """Format & tokenize it so it is ready for training
        :param tokenizer (AutoTokenizer): Model Tokenizer
        :param max_length (int): Maximum number of tokens to emit from tokenizer
        """
        
        # Add prompt to each sample
        print("Preprocessing dataset...")
        dataset = dataset.map(create_prompt_formats,)#, batched=True)
        
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            batch_size=32,
            remove_columns=['question', 'answer'],
        )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

        return dataset