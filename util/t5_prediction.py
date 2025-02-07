import sys
import os
# 将父目录添加到系统路径，便于模块导入
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from typing import Tuple, List

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from util.log_util import Log

# 添加中文注释以提高可读性，重命名部分变量以增强代码可读性

def generate_model_predictions(
        model, 
        dataloader, 
        tokenizer, 
        device, 
        logger: Log = None,
        return_score: bool = False,
        special_handling: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
    """
    生成模型的预测结果，并返回文本、预测和实际标签的列表。
    
    Args:
        model: 训练好的模型。
        dataloader (DataLoader): 包含测试数据的数据加载器。
        tokenizer (PreTrainedTokenizer): 用于分词的分词器。
        device: 模型和数据在哪个设备上执行（例如 'cpu' 或 'cuda'）。
        logger (Log, optional): 用于记录信息的日志工具。
        return_score (bool, optional): 是否返回性能分数（精度、召回率和F1）。
        special_handling (bool, optional): 是否进行特殊处理，例如处理特殊格式的标签。
    
    Returns:
        Tuple[List[str], List[str], List[str]]: 文本、预测和实际标签列表。
        如果 return_score 为 True，则还会返回平均损失、精度、召回率和 F1 分数。
    
    Examples:
        >>> texts, predictions, actuals = generate_model_predictions(model, dataloader, tokenizer, 'cpu')
        >>> print(predictions[0], actuals[0])
    """
    model.eval()  # 设置模型为评估模式
    if logger:
        logger.log_info("模型已设置为评估模式。")
    
    texts, predictions, actual_labels = [], [], []
    total_loss = 0
    
    # 遍历数据加载器中的每个批次
    for batch in dataloader:
        # 将数据移动到指定的设备上（例如 GPU 或 CPU）
        attention_mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        text_data = batch['text']
        
        with torch.no_grad():  # 不计算梯度，节省计算资源
            # 前向传播以获取模型输出
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels)
            
            # 计算损失
            loss = outputs.loss
            total_loss += loss.item()  # 累加损失
            
            # 使用生成方法获得模型的预测结果
            generated_outputs = model.generate(input_ids=input_ids, max_length=250, num_beams=5, early_stopping=True)
            decoded_predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_outputs]
            decoded_labels = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
        
        # 将文本、预测和实际标签添加到列表中
        texts.extend(text_data)
        predictions.extend(decoded_predictions)
        actual_labels.extend(decoded_labels)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 计算评价指标（精度、召回率和 F1 分数）
    precision = precision_score(actual_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(actual_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(actual_labels, predictions, average='macro', zero_division=0)
    
    # 如果需要进行特殊处理
    if special_handling:
        actual_labels = [label.split('_')[0] for label in actual_labels]
        predictions = [prediction.split('_')[0] for prediction in predictions]
        precision = precision_score(actual_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(actual_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(actual_labels, predictions, average='macro', zero_division=0)
    
    # 记录日志信息
    if logger:
        logger.log_info(f"平均损失: {avg_loss}")
        logger.log_info(f"精度: {precision}")
        logger.log_info(f"召回率: {recall}")
        logger.log_info(f"F1 分数: {f1}")
    
    # 根据需求返回相应的结果
    if return_score:
        return texts, predictions, actual_labels, avg_loss, precision, recall, f1
    else:
        return texts, predictions, actual_labels