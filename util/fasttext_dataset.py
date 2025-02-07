# coding=utf-8
import sys
import os
# 将当前脚本的父目录加入到系统路径，以便引入其他模块
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import pandas as pd
from typing import List


def prepare_fasttext_dataset(dataframe: pd.DataFrame) -> List[str]:
    """
    准备用于FastText的数据集，将标签转换为FastText要求的格式，并清理文本中的换行符。
    
    Args:
        dataframe (pd.DataFrame): 包含文本数据和标签的数据框，要求至少包含 'label' 和 'text' 两列。

    Returns:
        List[str]: 符合FastText格式的字符串列表，每个字符串为 "__label__标签 文本内容"。

    示例：
        >>> df = pd.DataFrame({'label': ['positive', 'negative'], 'text': ['good product', 'bad experience']})
        >>> prepare_fasttext_dataset(df)
        ['__label__positive good product', '__label__negative bad experience']
    """
    # 将标签列转换为字符串类型
    dataframe['label'] = dataframe['label'].astype(str)

    # 将标签格式转换为符合FastText要求的格式（前缀为"__label__"）
    dataframe['label'] = dataframe['label'].apply(
        lambda label: '__label__' + label.replace(' ', '_') if not label.startswith('__label__') else label
    )

    # 清理文本中的换行符，以保证每个文本是一行数据
    dataframe['text'] = dataframe['text'].apply(lambda text: text.replace('\n', ' '))

    # 创建符合FastText格式的数据集列表，每个元素包含标签和文本
    fasttext_dataset = [f'{label} {text}' for label, text in zip(dataframe['label'], dataframe['text'])]

    return fasttext_dataset

# 示例调用
if __name__ == "__main__":
    sample_data = {
        'label': ['positive', 'negative', 'neutral'],
        'text': ['This is a great product!', 'I had a terrible experience.', 'It was okay, nothing special.']
    }
    df = pd.DataFrame(sample_data)
    fasttext_data = prepare_fasttext_dataset(df)
    for line in fasttext_data:
        print(line)  # 打印每一行FastText数据