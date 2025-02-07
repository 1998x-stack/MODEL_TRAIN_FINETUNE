# coding=utf-8
import sys
import os
# 将上级目录添加到系统路径，以便导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import fasttext
from util.log_util import Log
from util.text_util import evaluate_and_save_results_fasttext
from util.fasttext_dataset import prepare_fasttext_dataset


def fasttext_training_process(
        model_type: str,
        logger: Log = None,
        autotune_duration: int = 120,
    ):
    """
    FastText 模型训练过程

    参数:
    - model_type: str - 模型的类型，用于区分不同的训练数据
    - logger: Log - 日志记录器
    - autotune_duration: int - 自调时长，默认120秒

    返回:
    - model: fasttext.FastText._FastText - 训练好的FastText模型
    """
    # 获取训练数据的文件路径
    training_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/train.tsv')
    )
    # 读取训练数据库
    training_df = pd.read_csv(training_file_path, sep='\t')
    # 去除空值数据
    training_df.dropna(inplace=True)
    training_df.reset_index(drop=True, inplace=True)
    # 准备FastText训练数据
    train_dataset = prepare_fasttext_dataset(training_df)

    # 获取测试数据的文件路径
    testing_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/test.tsv')
    )
    # 读取测试数据库
    testing_df = pd.read_csv(testing_file_path, sep='\t')
    # 准备FastText测试数据
    test_dataset = prepare_fasttext_dataset(testing_df)

    # 日志记录测试数据和训练数据
    if logger:
        logger.log_info(f"Training data prepared from {training_file_path}.")
        logger.log_info(f"Testing data prepared from {testing_file_path}.")

    # 获取FastText所需的训练和测试数据文件路径
    fasttext_train_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/train.txt')
    )
    fasttext_test_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'data/{model_type}/test.txt')
    )

    # 如果文件存在，删除这两个文件
    if os.path.exists(fasttext_train_file_path):
        os.remove(fasttext_train_file_path)
    if os.path.exists(fasttext_test_file_path):
        os.remove(fasttext_test_file_path)

    # 将训练数据写入到FastText所需的文件中
    with open(fasttext_train_file_path, "w") as train_file:
        for item in train_dataset:
            train_file.write(f"{item}\n")

    # 将测试数据写入到FastText所需的文件中
    with open(fasttext_test_file_path, "w") as test_file:
        for item in test_dataset:
            test_file.write(f"{item}\n")

    # 记录训练数据和测试数据的文件路径
    if logger:
        logger.log_info(f"Training data saved to {fasttext_train_file_path}.")
        logger.log_info(f"Testing data saved to {fasttext_test_file_path}.")

    # 开始训练FastText模型
    model = fasttext.train_supervised(
        input=fasttext_train_file_path,
        autotuneValidationFile=fasttext_test_file_path,
        autotuneDuration=autotune_duration,
    )

    # 模型测试返回结果
    result = model.test(fasttext_test_file_path)
    if logger:
        logger.log_info(f"Model: {model_type} Precision: {result[1]} Recall: {result[2]}")

    # 保存模型到指定路径
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'models/saved_models/fasttext/{model_type}.bin')
    )
    model.save_model(model_path)
    if logger:
        logger.log_info(f"Model saved: {model_path}")

    # 测试并记录预测结果
    predict_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'verification_data/fasttext/{model_type}_predict.csv')
    )
    evaluate_and_save_results_fasttext(model, fasttext_test_file_path, predict_file_path)

    if logger:
        logger.log_info(f"Predictions saved to CSV: {predict_file_path}")

    return model