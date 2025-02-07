# coding=utf-8
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import pandas as pd
from tqdm import trange
from sklearn.model_selection import train_test_split
import torch
from transformers import T5Tokenizer, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
from util.log_util import Log
from util.t5_prediction import generate_model_predictions
from util.t5_dataset import TextToTextDataset

def finetune_t5(
        model_type: str,
        language: str = 'zh', 
        device_ids: str = '3', 
        default_size: str = 'small',
        num_epochs: int = 20,
        batch_size: int = 16,
        lr: float = 3e-5,
        logger: Log = None,
    ) -> torch.nn.Module:
    """
    Fine-tunes a T5 model on a given dataset.

    Args:
        model_type (str): The type of model to finetune.
        language (str): Language of the tokenizer (default is 'zh').
        device_ids (str): Comma-separated list of GPU device IDs to use.
        default_size (str): Size of the base model (e.g., 'small').
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        logger (Log): Logger for outputting information.

    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if logger:
            logger.log_info(f"GPU is available, using device {device_ids}.")
        torch.cuda.empty_cache()
        # 选择活动设备
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        torch.backends.cuda.matmul.allow_tf32 = True  # 使用TF32加速
        
    # 加载模型和分词器
    default_tokenizer_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__) + '/..', 
            f'models/basis_models/{default_size}-{language}-tokenizer'
        )
    )
    tokenizer = T5Tokenizer.from_pretrained(default_tokenizer_path)
    if logger:
        logger.log_info(f"Loading T5Tokenizer from {default_tokenizer_path}.")
    
    # 初始化 T5 模型
    default_model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__) + '/..', 
            f'models/saved_models/t5/{model_type}.pt'
        )
    )
    model = torch.load(default_model_path, map_location='cpu')
    model.to(device)
    
    if torch.cuda.is_available() and ',' in device_ids:
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in device_ids.split(',')])

    # 加载数据集
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__) + '/..', 
            f'data/{model_type}/finetune.tsv'
        )
    )
    data = pd.read_csv(data_path, sep='\t')
    train_data, test_data = train_test_split(data, test_size=0.3)
    train_dataset = TextToTextDataset(train_data, tokenizer)
    test_dataset = TextToTextDataset(test_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 配置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.train()
    min_avg_loss = float('inf')
    save_texts = []
    save_predictions = []
    save_actuals = []

    # 开始训练循环
    for epoch in trange(num_epochs, desc='Training Epochs'):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播，计算损失
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
        
        # 记录训练损失
        if logger:
            logger.log_info(f'Epoch {epoch + 1} | Loss: {total_loss / len(train_dataloader):.4f}')
        
        # 生成模型预测并评估
        texts, predictions, actuals, avg_loss, precision, recall, f1 = generate_model_predictions(
            model, test_dataloader, tokenizer, device, logger=logger, is_return_score=True, is_special=False
        )
        
        # 保存最优模型
        if min_avg_loss > avg_loss:
            min_avg_loss = avg_loss
            save_texts = texts
            save_predictions = predictions
            save_actuals = actuals
            best_model_save_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__) + '/..', 
                    f'models/saved_models/t5/{model_type}_finetune_best.pt'
                )
            )
            torch.save(model.state_dict(), best_model_save_path)
            if logger:
                logger.log_info(f"Best model saved to {best_model_save_path}")
    
    # 保存测试结果
    test_results_df = pd.DataFrame(
        {
            'text': save_texts, 
            'predictions': save_predictions, 
            'actuals': save_actuals,
        }
    )
    test_results_file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__) + '/..', 
            f'verification_data/t5/{model_type}_finetune_test_results.csv'
        )
    )
    test_results_df.to_csv(test_results_file_path, index=False)
    if logger:
        logger.log_info(f"Test results saved to {test_results_file_path}")
    
    return model