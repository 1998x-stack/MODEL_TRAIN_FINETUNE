import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import re, requests, json
from util.log_util import Log
from config.config import HEADERS
def gpt_response(
        prompt, 
        logger:Log=None, 
        model="gpt-3.5-turbo-16k", 
        tag="query_rewrite",
    ):
    try:
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        if logger:
            logger.log_info(f"length of prompt: {len(prompt)}", print_screen=True)
        else: 
            print(f"length of prompt: {len(prompt)}")
        temp_data = {
            "serviceCode": "SI_KG_PRODUCE",
            "model": model,
            "text": prompt,
            "tag": tag,
            "keyType":"L2000",
        }
        ret = requests.post("https://ai-platform-cloud-proxy.zhihuishu.com/chat-llm/v1/completions/common",
                            data=json.dumps(temp_data), headers=HEADERS)
        if 'data' in ret.json().keys():
            if logger:
                logger.log_info("GPTResponse: " + ret.json()['data'], print_screen=True)
            else:
                print("GPTResponse: " + ret.json()['data'])
            return ret.json()['data']
        else:
            if logger:
                logger.log_info("GPTResponse: " + ret.json()['message'], print_screen=True)
            else:
                print("GPTResponse: " + ret.json()['message'])
            return None
    except Exception as e:
        if logger:
            logger.log_exception()
        else:
            print(e)
        return None