import os
import sys
# 将上级目录添加到系统路径，以便导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from util.log_util import Log
from util.fasttext_dataset import prepare_fasttext_dataset
from util.text_util import evaluate_and_save_results_fasttext


def finetune_fasttext(model_type: str, logger: Log = None):
    """
    对已有的 fastText 模型进行微调

    Args:
        model_type (str): 模型类型，用于指定模型和数据的路径
        logger (Log): 日志对象，用于记录日志信息

    Returns:
        model: 训练好的 fastText 模型
    """
    # 定义路径
    model_path = generate_path('models/saved_models/fasttext', f'{model_type}.bin')
    data_path = generate_path('data', model_type, 'finetune.tsv')

    # 检查模型和数据文件是否存在
    validate_file_exists(model_path, "模型文件不存在")
    validate_file_exists(data_path, "训练数据文件不存在")

    if logger:
        logger.log_info(f"找到模型文件: {model_path}")
        logger.log_info(f"找到训练数据文件: {data_path}")

    # 加载和分割数据集
    data = pd.read_csv(data_path, sep='	')
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # 准备 fastText 数据集
    train_dataset = prepare_fasttext_dataset(train_data)
    test_dataset = prepare_fasttext_dataset(test_data)

    # 保存训练集和测试集
    train_data_path = generate_path('data', model_type, 'finetune_train.txt')
    test_data_path = generate_path('data', model_type, 'finetune_test.txt')
    save_dataset_to_file(train_dataset, train_data_path, logger, "训练数据")
    save_dataset_to_file(test_dataset, test_data_path, logger, "测试数据")

    # 训练模型
    output_model_path = generate_path('models/saved_models/fasttext', f'{model_type}_finetune.bin')
    model = train_fasttext_model(train_data_path, test_data_path, model_path, output_model_path, logger)

    # 保存预测结果
    predict_results_path = generate_path('verification_data/fasttext', f'{model_type}_finetune_predict.csv')
    evaluate_and_save_results_fasttext(model, test_data_path, predict_results_path)
    if logger:
        logger.log_info(f"预测结果已保存至 {predict_results_path}")

    return model

def generate_path(*path_parts: str) -> str:
    """
    生成绝对路径

    Args:
        *path_parts (str): 路径的部分组成

    Returns:
        str: 拼接后的绝对路径
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', *path_parts))

def validate_file_exists(file_path: str, error_message: str):
    """
    验证文件是否存在，不存在则抛出异常

    Args:
        file_path (str): 文件路径
        error_message (str): 错误信息
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{error_message}: {file_path}")

def save_dataset_to_file(dataset: list, file_path: str, logger: Log, data_description: str):
    """
    将数据集保存到文件中

    Args:
        dataset (list): 数据集
        file_path (str): 文件路径
        logger (Log): 日志对象
        data_description (str): 数据集描述信息（用于日志记录）
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:  # 使用utf-8编码进行写入，防止文本格式问题
            for item in dataset:
                file.write(f"{item}\n")
        if logger:
            logger.log_info(f"{data_description}已保存至 {file_path}")
    except Exception as e:
        if logger:
            logger.log_error(f"保存{data_description}至 {file_path} 时出错: {str(e)}")
        raise

def train_fasttext_model(train_file_path: str, validation_file_path: str, pretrained_vectors_path: str, 
                         output_model_path: str, logger: Log):
    """
    训练 fastText 模型并保存

    Args:
        train_file_path (str): 训练数据文件路径
        validation_file_path (str): 验证数据文件路径
        pretrained_vectors_path (str): 预训练向量文件路径
        output_model_path (str): 输出模型保存路径
        logger (Log): 日志对象

    Returns:
        model: 训练好的 fastText 模型
    """
    try:
        model = fasttext.train_supervised(
            input=train_file_path,
            autotuneValidationFile=validation_file_path,
            pretrainedVectors=pretrained_vectors_path,
            epoch=25,
            lr=1.0,
        )
        model.save_model(output_model_path)
        result = model.test(validation_file_path)  # 用测试集进行模型测试
        precision = result.precision
        recall = result.recall
        if logger:
            logger.log_info(f"模型: Precision: {precision} Recall: {recall}")
            logger.log_info(f"模型已保存至 {output_model_path}")
        return model
    except Exception as e:
        if logger:
            logger.log_error(f"训练或保存模型时出错: {str(e)}")
        raise