import json
from typing import Tuple, Dict, List
from datasets import load_dataset, concatenate_datasets, Dataset

class DatasetService:
    def __init__(self, save_dir: str = "./data/EmbeddModel_Finetuning") -> None:
        """
        Initializes the dataset service with a default directory to save processed datasets.
        """
        self.save_dir = save_dir

    def preprocess_qas(self, input_file: str, output_file: str) -> None:
        """
        Filters QA pairs with non-null question and answer, saves valid entries,
        and creates train/test splits.
        """
        valid_count = self._filter_valid_qas(input_file, output_file)
        dataset = self._prepare_dataset(output_file)
        self._save_dataset(dataset)
        print(f"Preprocessing complete. Valid QA pairs: {valid_count}")

    def load_dataset_with_structure(
        self, test_path: str, train_path: str
    ) -> Tuple[Dataset, Dataset, Dict[int, str], Dict[int, str], Dict[int, List[int]]]:
        """
        Loads the dataset and structures it for retrieval model training/evaluation.
        """
        test_dataset = load_dataset("json", data_files=test_path, split="train")
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

        corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
        queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
        relevant_docs = {qid: [qid] for qid in queries}

        return train_dataset, test_dataset, corpus, queries, relevant_docs

    # ---------------- Private helper methods ---------------- #

    def _filter_valid_qas(self, input_file: str, output_file: str) -> int:
        """
        Filters QA pairs with both question and answer fields.
        """
        total_count = 0
        valid_count = 0

        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for line in infile:
                total_count += 1
                data = json.loads(line)

                if data.get("question") and data.get("answer"):
                    valid_count += 1
                    outfile.write(json.dumps(data) + "\n")

        print(f"Total: {total_count}, Valid: {valid_count}, Removed: {total_count - valid_count}")
        return valid_count

    def _prepare_dataset(self, output_file: str):
        """
        Loads the filtered QA dataset and processes it (renaming, ID column, split).
        """
        dataset = load_dataset("json", data_files=output_file)
        dataset = dataset.rename_column("question", "anchor")
        dataset = dataset.rename_column("answer", "positive")
        dataset = dataset["train"].add_column("id", range(len(dataset["train"])))
        return dataset.train_test_split(test_size=0.1)

    def _save_dataset(self, dataset) -> None:
        """
        Saves the train and test datasets to disk.
        """
        dataset["train"].to_json(f"{self.save_dir}/train_dataset.json", orient="records")
        dataset["test"].to_json(f"{self.save_dir}/test_dataset.json", orient="records")