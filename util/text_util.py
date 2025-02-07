# coding=utf-8
import sys
import os
# 将父目录添加到系统路径，便于模块导入
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import csv
import json
from rouge_score import rouge_scorer
from lingua import Language, LanguageDetectorBuilder

# 构建支持英语和中文的语言检测器
supported_languages = [
    Language.ENGLISH, Language.CHINESE
]
language_detector = LanguageDetectorBuilder.from_languages(*supported_languages).build()

def detect_language(text: str) -> str:
    """
    检测给定文本的语言。

    参数:
    - text (str): 要检测的文本。

    返回:
    - str: 检测到的语言的ISO 639-1代码。
    """
    try:
        return language_detector.detect_language_of(text).iso_code_639_1.name
    except Exception as e:
        print(f"语言检测失败: {e}")
        return None

def extract_json_from_text(text: str) -> list[dict] | dict | None:
    """
    从给定的文本中提取JSON数据，处理多个JSON对象。

    参数:
    - text (str): 包含JSON数据的输入文本。

    返回:
    - list[dict] | dict | None: 提取的JSON数据列表或单个JSON对象，如果没有找到则返回None。
    """
    json_objects = []
    start_index = 0
    while start_index < len(text):
        # 查找潜在JSON结构的起始位置
        start_index = text.find('{', start_index)
        if start_index == -1:  # 如果没有找到更多的JSON对象，则退出循环
            break

        # 跟踪开放和关闭大括号的数量以确定JSON对象的结束位置
        open_braces_count = 0
        end_index = start_index
        while end_index < len(text):
            current_char = text[end_index]
            if current_char == '{':
                open_braces_count += 1
            elif current_char == '}':
                open_braces_count -= 1
            
            if open_braces_count == 0:
                break
            end_index += 1

        # 提取潜在的JSON内容
        potential_json = text[start_index:end_index + 1]

        # 检查是否为有效的JSON
        try:
            json_objects.append(json.loads(potential_json))
        except json.JSONDecodeError:
            print(f"无效的JSON数据: {potential_json}")
            pass

        # 更新起始索引以查找下一个JSON对象
        start_index = end_index + 1

    if not json_objects:
        return None

    return json_objects if len(json_objects) > 1 else json_objects[0]

def evaluate_and_save_results_fasttext(model, test_file_path: str, result_file_path: str) -> None:
    """
    评估FastText模型并将测试结果保存到文件。

    参数:
    - model: FastText模型实例。
    - test_file_path (str): 测试文件路径，包含待评估的数据。
    - result_file_path (str): 结果文件路径，用于保存模型预测结果。
    """
    try:
        with open(test_file_path, 'r', encoding='utf-8') as test_file, \
             open(result_file_path, 'w', newline='', encoding='utf-8') as result_file:
            
            csv_reader = csv.reader(test_file)
            csv_writer = csv.writer(result_file)

            # 写入结果文件的表头
            csv_writer.writerow(["text", "predicted_label", "actual_label"])

            # 遍历测试文件中的每一行数据
            for row in csv_reader:
                # 从行中提取文本内容，移除标签部分
                text = " ".join(row[0].split()[1:])
                # 使用模型预测标签
                predicted_label, _ = model.predict(text)
                # 获取实际的标签
                actual_label = row[0].split()[0]
                # 将文本、预测标签和实际标签写入结果文件
                csv_writer.writerow([text, predicted_label[0], actual_label])
    except Exception as e:
        print(f"评估过程中发生错误: {e}")

def calculate_rouge_scores(reference_text: str, generated_text: str) -> dict:
    """
    计算参考文本和生成文本之间的ROUGE分数。

    参数:
    - reference_text (str): 参考文本。
    - generated_text (str): 生成的文本。

    返回:
    - dict: 包含ROUGE分数的字典。
    """
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference_text, generated_text)
    return rouge_scores

if __name__ == '__main__':
    # 示例：计算ROUGE分数
    example_scores = calculate_rouge_scores(
        'The quick brown fox jumps over the lazy dog',
        'The quick brown dog jumps on the log.'
    )
    print(f"ROUGE分数: {example_scores}")