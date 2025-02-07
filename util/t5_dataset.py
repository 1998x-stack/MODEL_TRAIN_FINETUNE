# coding=utf-8
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch
import pandas as pd
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
class TextToTextDataset(Dataset):
    """
    为文本到文本模型（如T5）创建一个PyTorch数据集。
    Args:
        dataframe (pd.DataFrame): 包含文本和标签的数据帧。
        tokenizer (PreTrainedTokenizer): 预训练的分词器。
        max_length (int): 文本的最大长度。
        label_max_length (int): 标签的最大长度。
    Attributes:
        tokenizer (PreTrainedTokenizer): 分词器实例。
        data (pd.DataFrame): 输入的数据帧。
        max_length (int): 文本编码的最大长度。
        label_max_length (int): 标签编码的最大长度。
    Examples:
        >>> from transformers import T5Tokenizer
        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> dataframe = pd.DataFrame({'text': ['hello world'], 'label': ['greet']})
        >>> dataset = TextToTextDataset(dataframe, tokenizer)
        >>> len(dataset)
        1
        >>> dataset[0]  # 实际输出将包括编码的input_ids和attention_mask
        {'text': 'hello world', ...}
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_length: int = 500, label_max_length: int = 250):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.label_max_length = label_max_length
        self.is_numeric_dtype = pd.api.types.is_numeric_dtype(dataframe.iloc[0, 1])
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        text = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1]
        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对标签进行编码
        # 如果labels是数值型数据，则直接转换为torch.Tensor
        if self.is_numeric_dtype:
            label_encoding = torch.tensor([labels], dtype=torch.long)
        else:
            # 非数值型标签，使用tokenizer.encode_plus处理
            label_encoding = self.tokenizer.encode_plus(
                labels,
                add_special_tokens=True,
                max_length=self.label_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label_encoding = label_encoding['input_ids'].squeeze()
        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_encoding,
        }