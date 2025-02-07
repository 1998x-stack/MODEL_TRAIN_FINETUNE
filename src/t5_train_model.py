# coding=utf-8
import sys
import os
# 将上级目录添加到系统路径，以便导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from tqdm import tqdm, trange
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from util.log_util import Log
from util.t5_dataset import TextToTextDataset
from util.t5_prediction import generate_model_predictions

# TODO: use multi-device for parallel training

def train_t5_model(
        model_type: str,
        language: str = 'zh',
        device_ids: str = '3',
        model_size: str = 'small',
        num_epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        logger: Log = None
    ) -> T5ForConditionalGeneration:
    """
    这个函数用于训练 T5 模型

    参数:
    model_type: 模型类型名称
    language: 语言类型 ('zh' 为中文)
    device_ids: 运行模型的 GPU ID
    model_size: 模型应用的规模大小
    num_epochs: 训练轮数
    batch_size: 训练批量大小
    learning_rate: 学习率
    logger: 日志对象

    返回:
    训练好的 T5 模型
    """
    # 设置运行设备 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if logger:
            logger.log_info(f"GPU is available, using device {device_ids}.")
        torch.cuda.empty_cache()  # 清除 GPU 内存
        
        # 设置 GPU 设备和 CUDA 变量
        if isinstance(device_ids, int):
            torch.cuda.set_device(device=device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
        elif isinstance(device_ids, str):
            default_device_id = int(device_ids.split(',')[0])
            torch.cuda.set_device(device=default_device_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        
        torch.backends.cuda.matmul.allow_tf32 = True  # 激活 TF32 用于 GPU 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 设置层分大小

    # 加载模型和分词器
    default_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'models/basis_models/{model_size}-{language}-model'))
    default_tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'models/basis_models/{model_size}-{language}-tokenizer'))
    
    if logger:
        logger.log_info(f"加载模型从 {default_model_path}.")
    model = T5ForConditionalGeneration.from_pretrained(default_model_path)
    model.to(device)
    
    if torch.cuda.is_available() and (isinstance(device_ids, str) and len(device_ids) > 1):
        # 如果用于多设备训练，将模型展开到多个 GPU
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in device_ids.split(',')])
    
    if logger:
        logger.log_info(f"加载 T5Tokenizer 从 {default_tokenizer_path}.")
    tokenizer = T5Tokenizer.from_pretrained(default_tokenizer_path)
    
    # 设置参数
    learning_rate = learning_rate  # 训练训练的学习率
    
    # 创建数据加载器
    training_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/train.tsv'))
    testing_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/test.tsv'))
    
    if logger:
        logger.log_info(f"加载训练数据从 {training_file_path}.")
    assert os.path.exists(training_file_path), f"File not found: {training_file_path}"
    df_train = pd.read_csv(training_file_path, sep='	', engine='python')
    
    if logger:
        logger.log_info(f"加载测试数据从 {testing_file_path}.")
    assert os.path.exists(testing_file_path), f"File not found: {testing_file_path}"
    df_test = pd.read_csv(testing_file_path, sep='	', engine='python')
    
    # 创建数据集
    dataset_train = TextToTextDataset(df_train, tokenizer)
    dataset_test = TextToTextDataset(df_test, tokenizer)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 设置学习率调度器
    total_steps = len(dataloader_train) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 记录最佳模型的变量
    min_avg_loss = float('inf')
    save_texts, save_predictions, save_actuals = [], [], []
    
    # 训练模型
    for epoch in trange(num_epochs, desc="Training Epochs"):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader_train, total=len(dataloader_train), desc="Training Batches"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader_train)
        if logger:
            logger.log_info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
        
        texts, predictions, actuals, avg_loss, precision, recall, f1 = generate_model_predictions(model, dataloader_test, tokenizer, device, logger=logger, is_return_score=True, is_special=False)
        
        # 保存最佳模型
        if avg_loss < min_avg_loss:
            min_avg_loss = avg_loss
            save_texts, save_predictions, save_actuals = texts, predictions, actuals
            best_model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'models/saved_models/t5/{model_type}_{epoch % 2}.pt'))
            torch.save(model.state_dict(), best_model_save_path)
            if logger:
                logger.log_info(f"Best model saved to {best_model_save_path}")

    # 保存预测结果
    df_predict = pd.DataFrame({'text': save_texts, 'predictions': save_predictions, 'actuals': save_actuals})
    predict_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'verification_data/t5/{model_type}_predict.csv'))
    df_predict.to_csv(predict_file_path, index=False, escapechar='\\')
    if logger:
        logger.log_info(f"Predictions saved to {predict_file_path}")
    
    # 打印前5个预测结果验证
    for i in range(5):
        if logger:
            logger.log_info(f"Prediction: {save_predictions[i]}")
            logger.log_info(f"Actual: {save_actuals[i]}")
    
    return model