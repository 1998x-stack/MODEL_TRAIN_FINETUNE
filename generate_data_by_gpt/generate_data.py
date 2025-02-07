import sys
import os
# 将上级目录添加到模块搜索路径
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import pandas as pd
from util.gpt_utils import gpt_response
from config.config import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

# 此为模板实例化，用于生成提示模板
prompt_template = PromptTemplate()

def generate_search_prompt(input_text: str, prompt_template: str) -> str:
    """
    根据输入文本和模板生成搜索提示。
    
    Args:
        input_text (str): 输入文本。
        prompt_template (str): 模板字符串。
        
    Returns:
        str: 生成的搜索提示。
    """
    print(f'正在生成提示：{input_text}')
    prompt = prompt_template.replace('$input$', input_text)
    return gpt_response(prompt)

def update_dataframe_with_gpt_response(dataframe: pd.DataFrame, prompt_template: str) -> pd.DataFrame:
    """
    使用多线程更新数据框中的GPT应答。
    
    Args:
        dataframe (pandas.DataFrame): 待更新的数据框。
        prompt_template (str): 用于生成提示的模板。
        
    Returns:
        pandas.DataFrame: 更新后的数据框。
    """
    # 根据‘text’列去重
    dataframe = dataframe.groupby('text').first().reset_index()
    dataframe.reset_index(drop=True, inplace=True)
    dataframe['gpt_response'] = None
    
    # 使用多线程进行应答生成
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(generate_search_prompt, text, prompt_template): idx
            for idx, text in enumerate(dataframe['text'])
        }
        
        # 处理完成的线程
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'文本 "{dataframe.loc[idx, "text"]}" 生成应答时出现异常: {exc}')
            else:
                dataframe.at[idx, 'gpt_response'] = result
    
    # 将更新后的数据框保存到CSV文件中
    dataframe.to_csv('updated_dataframe.csv', index=False)
    return dataframe

def update_valid_data():
    """
    更新有效数据文件中的数据，并生成GPT应答。
    """
    dataframe = pd.read_csv('data/0_origin_data_by_date/valid.csv', engine='python')
    updated_dataframe = update_dataframe_with_gpt_response(dataframe, prompt_template.CHANNEL_PROMPT)
    updated_dataframe.to_csv('data/updated_valid_dataframe.csv', index=False)

if __name__ == '__main__':
    # update_valid_data()
    pass