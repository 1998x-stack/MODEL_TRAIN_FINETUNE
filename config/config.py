# coding=utf-8
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


HEADERS = {
    "Content-Type": 'application/json',
    "service-code": 'SI_KG_PRODUCE'
}
class PromptTemplate:
    def __init__(self) -> None:
        
        self.IsValid_PROMPT = """
        ---
        **任务描述**：
        根据给定的文本，使用以下详细规则和标准，判断文本是否符合合格标准，并给出具体的理由。
        **合格标准**：
        1. **涉及专业知识**：文本需包含专业性词汇，或在作答时需要结合学术资源、法律资源、教育资源等，如医药、工程技术、法律等领域。
        2. **有明显检索意图**：文本表现出对特定信息的明确检索需求。
        3. **涉及道德判断、政治立场**：文本中包含道德或政治方面的立场或判断。
        **不合格标准**：
        1. **常识性问题**：内容主要关于常识性知识。
        2. **个人咨询**：如个人情感、生活建议、职业规划等咨询。
        3. **特定任务**：要求解决一道具体问题、编程任务、翻译任务等。
        4. **意图不明显**，且未提供足够信息以判断是否涉及专业知识。
        5. **意见表达&创作类**：如诗歌创作、心得体会、观后感等主观创作。除非该文本需基于专业知识回答，例如：大学生汽车构造实习心得（需基于汽车构造专业知识）；请写一篇文献综述，主题为智能化会计（需基于智能化会计知识）。
        **Few-shot 示例**：
        1. **问题**：解释什么是区块链技术。
        **答案**：
        ```
        {
            "是否合格":"是",
            "原因":"文本涉及专业知识，需要结合学术资源解答。"
        }
        ```
        2. **问题**：我今天感觉很郁闷，应该怎么办？
        **答案**：
        ```
        {
            "是否合格":"否",
            "原因":"文本为个人咨询，且属于常识性问题。"
        }
        ```
        **自我推理过程**：
        1. **检查是否包含专业性词汇**：确定文本中是否有需要专业背景知识才能理解或回答的词汇。
        2. **评估是否有明确的检索意图**：分析文本是否明确表达了寻求特定信息的意图。
        3. **识别是否涉及道德或政治立场**：判断文本中是否包含与道德或政治相关的立场或观点。
        **输出格式要求**：请按照以下Json格式精确输出，确保不输出其他内容。
        ```
        {
        "是否合格":"是"或"否",
        "原因":"<具体原因>"
        }
        ```
        **注意事项**：
        - 仔细分析文本内容，确保准确应用上述标准进行判断。
        - 对于含糊或信息不足的文本，需要谨慎评估其可能的含义和范畴。
        ---
        请根据上述规则和示例，判断以下文本是否合格，并给出您的判断依据：
        input: $input$
        output:
        """
        
        
        self.CHANNEL_PROMPT = """
        ---
        请根据以下规则，结合文本意图，给出文本input的推荐渠道。
        一个文本可以有多个意图，请考虑所有意图的可能性，但不要过度联想。请将一个文本作为一个整体判断，不要因为标点符号，对其中的每个短句进行判断。
        具体说明如下：
        - 【理论知识】想要了解专业相关理论知识具体内容，包括职中大学研究生院校的各类专业知识及课本内容，包括专业的专有名词，不包括“选择什么课程”“课程框架”“课程主要内容”“生活知识”“小说”“考试复习重点”。
        正确案例：遵义会议的历史意义是什么？为什么召开？、tRNA的基本概念和功能；
        错误案例：安全管理学主要内容
        - 【案例实验知识】想要了解实验的具体内容。出现“案例分析”“法条判例”时，可判断为此渠道。不包括“举几个例子”“举例说明”。
        正确案例：医学小白鼠实验，化学反应实验、计算机编程
        错误案例：葡萄糖跨膜物质转运的方式有哪些？请举例说明
        - 【科研知识】想要了解专业研究方向或前沿进展，阅读科研文献，撰写论文。当出现“文献”“前沿”“进展”“科研”词语时，判断为此渠道。
        正确案例：流体力学前沿问题？
        错误案例：RNA的转录在原核生物和真核生物中的异同
        - 【政策法律】想要了解法律法规和国家政策或运用相关信息解决问题。出现“诉讼”“中国特色社会主义”“法律”“二十大”，可判断为此渠道。
        如：张三李四的法律纠纷
        - 【题目查询】想要查找相关习题或相关题库。判断文本为填空题/选择题/概述题时，判断为此渠道。
            填空题：基尔霍夫定律仅适用于...电路。
            选择题：呼吸音减低, 腹部正常, 拟诊考虑为A. 胸膜间皮细胞瘤B. 大叶性肺炎C. 充血性心衰D. 结核性胸膜炎E. 肝硬化
            概述题：出现“简述”“试述"时。遵义会议的历史意义是什么？为什么召开？、常见的物业水景有哪些?
        - 【图片查询】想要获取相关图片。出现“图片”“照片”“img”“png"时，可判断为该渠道。
        如：液体安全阀结构图？
        - 【其他】：上述均不满足文本意图时，判断为此意图。出现“心得体会”“观后感”“建议”“课程大纲”“课程体系”“就业信息”词语时，可判断为此渠道。
        此外，请遵守下列硬性规则
        1.出现“含义”，”名词解释“，“xx和xx的区别”：判断意图为【理论知识】和【题目查询】
        2.出现“文献”，“前沿”，“进展”，“科研”：判断意图为【科研知识】
        3.当文本为一段文言文时：判断意图为【理论知识】
        注意事项：判断为【理论知识】、【案例实验知识】、【科研知识】、【政策法律】、【题目查询】、【图片查询】的情况下，则不会再判断为【其他】
        ---
        input: $input$
        output:
        """
        
        
        self.CHANNEL_SELECTOR_PROMPT = """
        ---
        **Objective**: Based on the provided text, select the most appropriate channel for information retrieval from the following options: 'academic resources', 'educational resources', 'legal resources', 'search engines', 'news and consultation', 'knowledge graph', and 'other resources'. It's important to note that the first five channels should be prioritized unless the text clearly indicates a better fit with another channel.
        **Instructions**:
        1. **Analyze the Input Text**: Carefully read the input text to understand its main theme and the type of information being sought.
        2. **Apply Reasoning**: Use a chain of thought approach to reason concisely through why a particular channel is the best fit based on the text's content.
        3. **Prioritize Channels**: Remember to give higher selection weight to 'academic resources', 'educational resources', 'legal resources', 'search engines', and 'news and consultation' unless the input strongly suggests another channel is more appropriate.
        4. **Make Your Selection**: Choose the most suitable information retrieval channel based on your analysis and reasoning.
        ### Example Prompts with Chain of Thought and Self-Reasoning
        - **Input Text**: "I'm looking for peer-reviewed studies on the impact of climate change on Arctic biodiversity."
            - **Thought Process**: This request is specifically asking for scholarly articles, which are best found in academic databases. Given the focus on 'peer-reviewed studies', the 'academic resources' channel is the most appropriate choice.
            - **Selected Channel**: Academic Resources
        - **Input Text**: "What are the latest legal precedents regarding copyright infringement in digital media?"
            - **Thought Process**: This inquiry requires access to legal documents and case law, pointing directly to the 'legal resources' channel. It's a specialized query that necessitates comprehensive legal databases.
            - **Selected Channel**: Legal Resources
        - **Input Text**: "I need a broad overview of World War II for my history class presentation."
            - **Thought Process**: This is an educational request suitable for a general audience, making 'educational resources' the best fit. The need for a broad overview suggests that academic journals might be too specific, whereas educational sites provide summarized and accessible information.
            - **Selected Channel**: Educational Resources
        ---
        input: $input$
        output:
        """
        
        
        self.IS_NEED_SEARCH_PROMPT = """
        ---
        **Objective**: This prompt is designed to assist ChatGPT in deciding whether an input query necessitates conducting an external search to provide an accurate and up-to-date response or if it can be adequately answered using its existing knowledge base.
        **Instructions**:
        1. **Introduction**:
        - "Your task is to assess each input query to determine if an external search is required for providing an accurate and comprehensive answer. This assessment should be based on predefined criteria that categorize queries into those that necessitate a search and those that do not."
        2. **Criteria for Decision-Making**:
        - *No Search Required*: Inputs requesting general knowledge explanations, definitions, simple conversational exchanges, or creative content (e.g., stories, poetry) should be addressed using the model's pre-trained knowledge, as these do not typically require up-to-the-minute information or highly specialized knowledge beyond what the model has been trained on.
        - *Search Required*: Inputs asking for the latest updates on news events, detailed explanations on specialized topics, specific factual information (e.g., recent statistics, event outcomes), or any query that likely extends beyond the model's pre-trained knowledge base should be flagged as requiring an external search.
        3. **Few-shot Examples with Reasoning**:
        - **Example 1**: Input: "Tell me a story about a dragon." - Decision: No search required. *Reasoning*: This request for creative content can be fulfilled using the model's inherent capabilities in generating imaginative and engaging stories.
        - **Example 2**: Input: "What is the latest update on the Mars rover mission?" - Decision: Search required. *Reasoning*: This query demands current information that may have evolved beyond the model's last training update, necessitating a search for the most recent developments.
        4. **Output Format**:
        - For each assessment, clearly state your decision using the format: "Search Required: [Yes/No]. Reasoning: [Brief explanation based on the criteria above]."
        5. **Instructions for Uncertainty**:
        - "If you encounter a query where the necessity for a search is unclear based on the provided criteria, please state your uncertainty and explain the factors that make the decision challenging."
        6. **Conclusion**:
        - "Your careful assessment and reasoned decision-making are crucial for ensuring that responses are accurate, relevant, and as informative as possible. Please adhere to the criteria and examples provided to guide your decisions on the necessity of conducting an external search."
        ---
        input: $input$
        output:
        """