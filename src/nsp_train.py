# coding=utf-8
import sys
import os
# 将上级目录添加到系统路径，以便导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForNextSentencePrediction, TrainingArguments, Trainer
from tensorboardX import SummaryWriter
from util.nsp_dataset import SentenceDataset

# 设置可见的GPU设备。这里选择0和1号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class BERTSentencePredictor:
    """
    一个处理BERT下一句语预测模型训练和请求的类。
    
    属性:
        device (torch.device): 运行模型的设备（CPU或GPU）。
        base_model_path (str): 基础预训练BERT模型的路径。
        save_model_path (str): 保存训练后BERT模型的路径。
    """
    def __init__(self):
        """
        初始化BERTSentencePredictor。
        """
        # 基础模型路径
        base_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/basis_models')
        # 选择运行模型的设备是CPU还是GPU
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # 中文BERT基础模型和远程保存路径
        self.zh_base_model_path = os.path.join(base_model_dir, 'bert-base-chinese')
        self.zh_save_model_path = os.path.join(base_model_dir, 'nsp_zh')
        # 英文BERT基础模型和远程保存路径
        self.en_base_model_path = os.path.join(base_model_dir, 'bert-base-uncased')
        self.en_save_model_path = os.path.join(base_model_dir, 'nsp_en')
        
    def load_dataset(self, csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载并将数据集拆分为训练集和测试集。
        
        参数:
            csv_path (str): 包含数据集的CSV文件的路径。
        
        返回:
            tuple: 包含训练集和测试集的元组。
        """
        # 使用pandas加载数据集
        df = pd.read_csv(csv_path, engine='python')
        # 分割数据集，还原的20%作为测试集
        train_df, test_df = train_test_split(df, test_size=0.2)
        # 重置训练集和测试集的index
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        return train_df, test_df
    
    def train_model(self, csv_path: str, lang: str = 'zh') -> None:
        """
        使用csv_path指定的数据集训练BERT模型。
        
        参数:
            csv_path (str): 用于训练的CSV文件的路径。
            lang (str): 语言（zh中文，en英文）。
        """
        # 基础模型路径
        base_model_path = self.zh_base_model_path if lang == 'zh' else self.en_base_model_path
        # 初始化汉语BERT分词器和下一句语预测模型
        tokenizer = BertTokenizer.from_pretrained(base_model_path)
        model = BertForNextSentencePrediction.from_pretrained(base_model_path)
        model.to(self.device)
        # 加载数据集
        train_df, test_df = self.load_dataset(csv_path)
        train_dataset = SentenceDataset(tokenizer, train_df)
        eval_dataset = SentenceDataset(tokenizer, test_df)
        # 训练参数设置
        training_args = TrainingArguments(
            output_dir=f'./nsp_{lang}/nsp_results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./nsp_{lang}/nsp_logs',
            logging_steps=10
        )
        # 创建Trainer对象
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        # 训练模型
        trainer.train()
        # 保存训练后的模型
        save_model_path = self.zh_save_model_path if lang == 'zh' else self.en_save_model_path
        trainer.save_model(save_model_path)
    
    def predict(self, sentence1: str, sentence2: str, lang: str) -> float:
        """
        预测sentence2是sentence1的下一句的概率。
        
        参数:
            sentence1 (str): 第一句话。
            sentence2 (str): 可能的下一句。
            lang (str): 语言（zh中文，en英文）。
        
        返回:
            float: sentence2是下一句的概率。
        """
        # 获取分词器和训练好的模型的路径
        tokenizer_path = self.zh_base_model_path if lang == 'zh' else self.en_base_model_path
        model_path = self.zh_save_model_path if lang == 'zh' else self.en_save_model_path
        # 初始化汉语BERT分词器和模型
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertForNextSentencePrediction.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        # 将两句话进行编码
        encoded_input = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            # 使用模型进行预测
            output = model(**encoded_input)
        # 计算概率
        probs = torch.softmax(output.logits, dim=1)
        return probs[0, 1].item()