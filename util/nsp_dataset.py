# coding=utf-8
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Dict
class SentenceDataset(Dataset):
    """
    Dataset class for BERT Next Sentence Prediction.
    This class handles the loading and processing of data for training the BERT model.
    """
    def __init__(self, tokenizer: BertTokenizer, data, max_length: int = 512) -> None:
        """
        Initializes the SentenceDataset class with a tokenizer, file path, and optional max length.
        
        Args:
            tokenizer (BertTokenizer): A tokenizer instance of the BERT tokenizer.
            filepath (str): Path to the CSV file containing the sentence pairs.
            max_length (int): Maximum length of tokenized sentence pairs.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an item by index.
        Args:
            idx (int): Index of the data item.
        
        Returns:
            A dictionary containing the necessary inputs for the BERT model, including input_ids, token_type_ids, attention_mask, and next_sentence_label.
        """
        # 获取数据行
        item = self.data.iloc[idx]
        sentence_a = item['text']
        sentence_b = item['label']
        label = int(item['next_sentence_label'])  # 此处假设标签名为 next_sentence_label 并且是0或1
        # Tokenization and encoding the pair
        encoded_pair = self.tokenizer(
            text=sentence_a,
            text_pair=sentence_b,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Squeeze to remove the batch dimension added by return_tensors
        encoded_pair = {key: val.squeeze(0) for key, val in encoded_pair.items()}
        encoded_pair['labels'] = torch.tensor([label], dtype=torch.long)
        
        return encoded_pair