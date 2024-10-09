import streamlit as st
import ollama
import re
import json
import plotly.express as px
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score
import pandas as pd
import plotly.graph_objects as go

###绘图###
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: score.fmeasure for key, score in scores.items()}

def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang='en', model_type='bert-base-uncased', verbose=True)
    return F1.mean().item()

def generate_metrics_plot(rouge_scores, bleu_score, bert_score):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore']
    scores = [
        rouge_scores['rouge1'],
        rouge_scores['rouge2'],
        rouge_scores['rougeL'],
        bleu_score,
        bert_score,
    ]
    
    df = pd.DataFrame({
        'Metrics': metrics,
        'Scores': scores
    })
    
    fig = px.bar(df, x='Metrics', y='Scores', text='Scores',
                 #title="Text Generation Evaluation Metrics",
                 color='Scores', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        xaxis_title='',  # 去掉x轴标题
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins to use more space for the chart
        height=350,
    )
    return fig


def plot_scores_PAIRWISE(processed_response, for_chart):
    multi_dimension_score = {
                        'score_A': processed_response['score_A'],
                        'score_B': processed_response['score_B']
                    }
    metrics = list(multi_dimension_score['score_A'].keys())
    scores_A = [multi_dimension_score['score_A'][metric] for metric in metrics]
    scores_B = [multi_dimension_score['score_B'][metric] for metric in metrics]
    
    # 创建 Plotly 图表
    fig1 = go.Figure(data=[
        go.Bar(name='Score A', x=metrics, y=scores_A),
        go.Bar(name='Score B', x=metrics, y=scores_B)
    ])
    
    # 更新图表布局
    fig1.update_layout(
        barmode='group',
        #title='Score Comparison for A vs B',
        #xaxis_title='Metric',
        yaxis_title='Scores',
        legend_title='Score Groups',
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )
    
    # 使用 Streamlit 的 st.plotly_chart() 来显示图表
    #st.plotly_chart(fig)

    #第一张和reference比较图
    reference = for_chart["reference"]
    candidate1 = for_chart["answer1"]
    rouge_scores = compute_rouge(reference, candidate1)
    bleu_score = sacrebleu.corpus_bleu([candidate1], [[reference]]).score
    bert_score_val = compute_bertscore(reference, candidate1)
    fig2 = generate_metrics_plot(rouge_scores, bleu_score/10, bert_score_val)

    #第二张和reference比较图
    reference = for_chart["reference"]
    candidate1 = for_chart["answer2"]
    rouge_scores = compute_rouge(reference, candidate1)
    bleu_score = sacrebleu.corpus_bleu([candidate1], [[reference]]).score
    bert_score_val = compute_bertscore(reference, candidate1)
    fig3 = generate_metrics_plot(rouge_scores, bleu_score/10, bert_score_val)

    return fig1, fig2, fig3
    

def plot_scores_POINTWISE(processed_response, for_chart):
    # 假设 processed_response 包含 'score_A' 或类似字段
    dimension_scores = processed_response["Dimension_Scores"]  # 使用 processed_response 作为输入数据
    
    metrics = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    # 创建 Plotly 图表
    fig1 = go.Figure(data=[
        go.Bar(x=metrics, y=scores)
    ])

    # 更新图表布局
    fig1.update_layout(
        #title='Dimension Scores',
        #xaxis_title='Metric',
        yaxis_title='Score',
        xaxis=dict(type='category'),  # 确保x轴为类别类型
        yaxis=dict(range=[0, max(scores) + 1]),  # 可以根据需要调整y轴范围
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )
    
    # 使用 Streamlit 的 st.plotly_chart() 方法显示图表
    #st.plotly_chart(fig)
     #第一张和reference比较图
    reference = for_chart["reference"]
    candidate1 = for_chart["answer"]
    rouge_scores = compute_rouge(reference, candidate1)
    bleu_score = (sacrebleu.corpus_bleu([candidate1], [[reference]]).score)/10
    bert_score_val = compute_bertscore(reference, candidate1)
    fig2 = generate_metrics_plot(rouge_scores, bleu_score, bert_score_val)

    return fig1,fig2


# Function to extract required parts from gpt_response
def extract_gpt_response_info_pairwise(gpt_response):
    # 正则表达式模式用于提取各部分
    pattern_a = r"@@@(.*?)@@@"
    pattern_b = r"@@@(.*?)###"
    pattern_final_result = r"###(.*?)&&&"
    pattern_detailed_feedback = r"&&&Detailed Evaluation Feedback:(.*?)\*\*\*"
    
    # 使用正则表达式提取各部分内容
    match_a = re.search(pattern_a, gpt_response, re.DOTALL)
    match_b = re.search(pattern_b, gpt_response, re.DOTALL)
    match_final_result = re.search(pattern_final_result, gpt_response, re.DOTALL)
    match_detailed_feedback = re.search(pattern_detailed_feedback, gpt_response, re.DOTALL)
    
    # 初始化结果字典
    result = {}

    # 对 dict_A 和 dict_B 使用字符串解析（非标准JSON格式无法直接解析）
    dict_a_raw = match_a.group(1).strip() if match_a else ""
    dict_b_raw = match_b.group(1).strip() if match_b else ""
    
    # 将自定义的格式转换为键值对字典
    def parse_custom_format(raw_text):
        scores = {}
        # 匹配类似 'Key': value 的格式
        matches = re.findall(r"'(.*?)':\s*(\d+)", raw_text)
        for key, value in matches:
            scores[key] = int(value)
        return scores
    
    # 解析 dict_A 和 dict_B
    result['score_A'] = parse_custom_format(dict_a_raw)
    result['score_B'] = parse_custom_format(dict_b_raw)
    result['final_results'] = match_final_result.group(1).strip() if match_final_result else ""
    result['Detailed_Evaluation_Feedback'] = match_detailed_feedback.group(1).strip() if match_detailed_feedback else ""
    
    return result

def extract_gpt_response_info_pointwise(gpt_response):
    # 正则表达式模式用于提取各部分
    pattern_dict_a = r"@@@Dimension Scores:\s*(\{.*?\})###"
    pattern_dict_b = r"###Overall Score:\s*(\d+)&&&"
    pattern_detailed_feedback = r"&&&Detailed Evaluation Feedback:(.*?)\*\*\*"

    # 使用正则表达式提取各部分内容
    match_dict_a = re.search(pattern_dict_a, gpt_response, re.DOTALL)
    match_dict_b = re.search(pattern_dict_b, gpt_response, re.DOTALL)
    match_detailed_feedback = re.search(pattern_detailed_feedback, gpt_response, re.DOTALL)

    # 初始化结果字典
    result = {}

    # 手动解析自定义格式的字典
    def parse_custom_format(raw_text):
        scores = {}
        # 匹配类似 'Key': value 的格式 (其中value是整数)
        matches = re.findall(r"'(.*?)':\s*(\d+)", raw_text)
        for key, value in matches:
            scores[key] = int(value)
        return scores

    # 解析字典A (Dimension Scores)
    dict_a_raw = match_dict_a.group(1).strip() if match_dict_a else ""
    dict_a = parse_custom_format(dict_a_raw)

    # 解析字典B (Overall Score)
    dict_b = {"Overall Score": int(match_dict_b.group(1).strip())} if match_dict_b else {}

    # 解析详细反馈 (Detailed Evaluation Feedback)
    detailed_feedback = match_detailed_feedback.group(1).strip() if match_detailed_feedback else ""

    # 将解析的内容存入结果字典
    result['Dimension_Scores'] = dict_a
    result['Overall_Score'] = dict_b
    result['Detailed_Evaluation_Feedback'] = detailed_feedback

    return result


def read_criteria(scenario):
    """根据场景读取相应的评价标准文本文件"""
    try:
        with open(f'./txt_simplify_criteria/{scenario}.txt', 'r', encoding='utf-8') as file:
            criteria = file.read()
        return criteria
    except FileNotFoundError:
        print(f"No criteria found for {scenario}")
        return ""

def user_selected_criteria(criteria_list):
    # 遍历列表，将每个元素转换为带有序号的格式
    formatted_criteria = [f"{i+1}. {criteria}" for i, criteria in enumerate(criteria_list)]
    # 将所有元素合并成一个字符串，每个元素占一行
    return "\n".join(formatted_criteria)
# App title
st.set_page_config(page_title="💬 EasyJudge",layout="wide")

model_list = ollama.list()

if "model_name" not in st.session_state:
    st.session_state["model_name"] = "PAIRWISE:latest"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title('💬 EasyJudge')
    st.write('EasyJudge is an innovative tool that uses large language models to precisely assess other models\' responses across multiple dimensions and various scenarios.')
    st.subheader('Models and parameters')
    option = st.selectbox(
        'Select a model',
        [model['name'].replace(':latest', '') for model in model_list['models']], index=1)
    st.write('You selected:', option)
    st.session_state["model_name"] = option
    st.divider()

    temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=1024, max_value=8192, value=1024, step=8)
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.clear()  # 清除 st.session_state 中的所有内容
    st.session_state['question_body'] = ""
    st.session_state['answer1_body'] = ""
    st.session_state['answer2_body'] = ""
    st.session_state['reference'] = ""
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


st.title(f"Judged by {st.session_state['model_name']} Model")

uploaded_file = st.file_uploader(
        "",
        type=["json", "jsonl"],
        help="Scanned documents are not supported yet!",
    )

options = ['default', 'analyzing_general', 'asking_how_to_question', 'brainstorming', 'chitchat', 'classification_identification', 'code_correction_rewriting', 'code_generation', 'code_to_code_translation', 'counterfactual', 'creative_writing', 'data_analysis', 'explaining_code', 'explaining_general', 'functional_writing', 'information_extraction', 'instructional_rewriting', 'keywords_extraction', 'language_polishing', 'math_reasoning', 'open_question', 'paraphrasing', 'planning', 'question_generation', 'ranking', 'reading_comprehension', 'recommendation', 'roleplay', 'seeking_advice', 'solving_exam_question_with_math', 'solving_exam_question_without_math', 'text_correction', 'text_simplification', 'text_summarization', 'text_to_text_translation', 'title_generation', 'topic_modeling', 'value_judgement', 'verifying_fact', 'writing_advertisement', 'writing_cooking_recipe', 'writing_email', 'writing_job_application', 'writing_news_article', 'writing_personal_essay', 'writing_presentation_script', 'writing_product_description', 'writing_social_media_post', 'writing_song_lyrics']

# 创建一个跟踪选择的变量
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = options[0]

def update_selection(option):
    st.session_state.selected_option = option

# 使用expander来组织显示，以便在需要时折叠和展开
with st.expander("Choose an scenario"):
    # 计算需要的行和列数量
    num_rows = len(options) // 8 + (1 if len(options) % 8 > 0 else 0)
    rows = [st.columns(8) for _ in range(num_rows)]
    option_index = 0

    for row in rows:
        for col in row:
            with col:
                # 仅在还有选项时显示单选按钮
                if option_index < len(options):
                    # 检查这个选项是否被选中
                    is_checked = st.radio(
                        "", [options[option_index]],
                        key=f"option_{option_index}",  # 确保每个单选按钮组的key不同
                        index=0 if st.session_state.selected_option == options[option_index] else None,
                        on_change=update_selection,
                        args=(options[option_index],)
                    )
                    option_index += 1

# 显示选中的选项
st.write("You selected:", st.session_state.selected_option)

# 定义维度评估复选框的选项
options_group_1 = [
    "text quality",
    "math correctness",
    "depth of understanding",
    "harmlessness",
    "information richness",
    "accuracy",
    "code correctness",
    "instruction following",
    "structure of answer",
    "Logical Reasoning"
]
options_group_2 = [
    "attractive",
    "variety",
    "concise",
    "professional",
    "writing style",
    "language style",
    "persuasive language",
    "vivid",
    "user-friendly",
    "tone"
]
options_group_3 = [
    "Clarity",
    "Relevance to Topic/Text",
    "Depth",
    "Coherence",
    "Originality",
    "Instruction Following",
    "Fluency",
    "Engagement",
    "Detail",
    "Creativity"
]
options_group_4 = [
    "Structure",
    "Readability",
    "Formatting and Layout",
    "Introduction",
    "Brief Summary at the Beginning",
    "Headline",
    "Length",
    "Hashtags",
    "signature",
    "visuals"
]

# 用于存储用户选择的选项，使用新的变量名 criteria_selected_option
if 'criteria_selected_option' not in st.session_state:
    st.session_state.criteria_selected_option = {
        "group_1": [],
        "group_2": [],
        "group_3": [],
        "group_4": []
    }

# 计算总的选项数量
total_selected = sum(len(st.session_state.criteria_selected_option[group]) for group in st.session_state.criteria_selected_option)

# 设置一个标志，超过 10 个选项时禁用复选框
disable_checkboxes = total_selected >= 10

# 创建一个包含 4 组复选框的横向排列
with st.expander("Select evaluation criteria"):
    # 使用 st.columns 创建 4 列布局
    cols = st.columns(4)

    # 在每列的顶部添加组名称
    with cols[0]:
        st.write("basic standard")
    with cols[1]:
        st.write("style")
    with cols[2]:
        st.write("content")
    with cols[3]:
        st.write("format")

    # 在每一列中放置复选框
    for i in range(10):
        # 第 1 组复选框
        with cols[0]:
            option = options_group_1[i]
            checked = option in st.session_state.criteria_selected_option["group_1"]
            if st.checkbox(option, key=f"group_1_{option}", value=checked, disabled=not checked and disable_checkboxes):
                if option not in st.session_state.criteria_selected_option["group_1"]:
                    st.session_state.criteria_selected_option["group_1"].append(option)
            else:
                if option in st.session_state.criteria_selected_option["group_1"]:
                    st.session_state.criteria_selected_option["group_1"].remove(option)

        # 第 2 组复选框
        with cols[1]:
            option = options_group_2[i]
            checked = option in st.session_state.criteria_selected_option["group_2"]
            if st.checkbox(option, key=f"group_2_{option}", value=checked, disabled=not checked and disable_checkboxes):
                if option not in st.session_state.criteria_selected_option["group_2"]:
                    st.session_state.criteria_selected_option["group_2"].append(option)
            else:
                if option in st.session_state.criteria_selected_option["group_2"]:
                    st.session_state.criteria_selected_option["group_2"].remove(option)

        # 第 3 组复选框
        with cols[2]:
            option = options_group_3[i]
            checked = option in st.session_state.criteria_selected_option["group_3"]
            if st.checkbox(option, key=f"group_3_{option}", value=checked, disabled=not checked and disable_checkboxes):
                if option not in st.session_state.criteria_selected_option["group_3"]:
                    st.session_state.criteria_selected_option["group_3"].append(option)
            else:
                if option in st.session_state.criteria_selected_option["group_3"]:
                    st.session_state.criteria_selected_option["group_3"].remove(option)

        # 第 4 组复选框
        with cols[3]:
            option = options_group_4[i]
            checked = option in st.session_state.criteria_selected_option["group_4"]
            if st.checkbox(option, key=f"group_4_{option}", value=checked, disabled=not checked and disable_checkboxes):
                if option not in st.session_state.criteria_selected_option["group_4"]:
                    st.session_state.criteria_selected_option["group_4"].append(option)
            else:
                if option in st.session_state.criteria_selected_option["group_4"]:
                    st.session_state.criteria_selected_option["group_4"].remove(option)


# 输出选中的名称
selected_criteria = []
for group in st.session_state.criteria_selected_option:
    selected_criteria.extend(st.session_state.criteria_selected_option[group])

# 检查是否有选中的标准
if 0 < len(selected_criteria) < 5:
    st.warning("You must select either 0 or at least 5 criteria.")
    disable_other_operations = True
    st.stop() 
else:
    disable_other_operations = False

# 只有当选中 0 个或 5 个及以上选项时，才允许执行其他操作
if not disable_other_operations:
    # 继续执行其他操作
    if selected_criteria:
        st.write(f"You selected: {', '.join(selected_criteria)}")  # 输出格式为 "You selected: ..."
    else:
        st.write("You selected: No criteria selected.")

#用户选择PAIRWISE
if st.session_state["model_name"] == "PAIRWISE":

    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False

    #用户选择PAIRSE，上传json文件
    if uploaded_file and not st.session_state.file_processed:
        try:
            # 重置文件指针到开始
            uploaded_file.seek(0)
            file_content = uploaded_file.read().decode('utf-8')  # 读取并解码为UTF-8格式的字符串
            data = json.loads(file_content)  # 解析JSON数据

            # 检查数据是否为列表
            if isinstance(data, list):
                # 检查每个元素是否包含必要的键
                missing_keys = []
                for i, item in enumerate(data):
                    required_keys = {'question_body', 'answer1_body', 'answer2_body'}
                    if not all(key in item for key in required_keys):
                        missing_keys.append(i)

                if not missing_keys:
                    st.success("All entries are correctly formatted.")
                    # 可以在这里处理数据
                else:
                    st.error(f"Missing required keys in entries: {missing_keys}")
                
                st.session_state.file_processed = True
            else:
                st.error("Uploaded file does not contain a JSON array.")
        except json.JSONDecodeError:
            st.error("The file is not a valid JSON file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # 尝试从本地文件系统加载模板文件
        template_file = "./prompt_template/PAIRWISE_WOR.txt"  # 更换为你的模板文件路径
        try:
            with open(template_file, "r") as file:
                base_prompt = file.read()
        except FileNotFoundError:
            st.error(f"Error: {template_file} file not found.")
            st.stop()

        # 处理上传的数据，并生成模型调用的prompt
        if isinstance(data, list):
            for item in data:
                if all(key in item for key in ['question_body', 'answer1_body', 'answer2_body', 'reference']):
                    question_body=item['question_body'],
                    answer1_body=item['answer1_body'],
                    answer2_body=item['answer2_body'],
                    reference = item['reference']
                    
                    # 在这里调用模型生成结果
                    # 示例：full_response = call_your_model(final_prompt)
                    final_prompt = base_prompt.format(
                        scenario = st.session_state.selected_option,
                        criteria = read_criteria(st.session_state.selected_option),
                        question_body=question_body,
                        answer1_body=answer1_body,
                        answer2_body=answer2_body,
                        reference= reference if reference else "N/A"  # Use "N/A" or any suitable placeholder if reference is empty
                    )

                    for_chart = {"reference":reference, "answer1":answer1_body, "answer2":answer2_body}

                    # Call ollama.chat to process the dialogue
                    #with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    results_holder = st.empty()

                    full_response = ""

                    with st.spinner('Runing...Please Wait...'):
                        start_output = False
                        # 获取模型响应的完整内容
                        for chunk in ollama.chat(
                            model=st.session_state["model_name"],
                            messages=[{"role": "user", "content": final_prompt}],
                            stream=True
                        ):
                            if 'message' in chunk and 'content' in chunk['message']:
                                full_response += (chunk['message']['content'] or "")
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                                # 一旦收到第一个内容块，结束加载提示（通过结束 `with` 自动消失）
                                if not start_output:
                                    start_output = True
                    message_placeholder.empty()

                    # 经过处理后再输出
                    processed_response = extract_gpt_response_info_pairwise(full_response)
                    

                    final_result = str(processed_response["final_results"]).replace("Final Result: ", "")
                    result_text = "🤝 It's a Tie!" if final_result == "Tie" else f"🏆 {final_result} Wins!"

                    # 使用Streamlit的columns进行布局优化
                    col1, col2 = st.columns([1, 3])

                    # 更新后的样式
                    common_style = "padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                    background_color1 = "background-color: #f8f9fa;"  # 浅灰色
                    background_color2 = "background-color: #e9ecef;"  # 较深灰色

                    with col1:
                        st.markdown(f"""
                            <div style="{background_color1} {common_style}">
                                <h2 style="color: #007BFF;">{result_text}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style="{background_color2} {common_style}">
                                <h3 style="color: #6c757d;">Detailed Evaluation Feedback</h3>
                                <p style="font-size: 16px; line-height: 1.6;">
                                    {processed_response["Detailed_Evaluation_Feedback"]}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # # 添加分隔符
                    # st.markdown("<hr>", unsafe_allow_html=True)
                    # 添加分隔符，设置样式减少上下间距
                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)  # 自定义上下边距


                    # 显示结果
                    #st.write(full_response)

                    # 将结果添加到原数据中
                    item['model_critic'] = full_response
                    st.write("successfully written to the JSON file")

            # 保存修改后的数据并提供下载链接
            modified_file_path = 'critic_by_pairwise_data.json'
            with open(modified_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            # 提供文件下载
            with open(modified_file_path, 'rb') as f:
                #st.download_button('Download critic_by_pairwise_data JSON', f, file_name=modified_file_path)
            
                if st.download_button('Download the evaluation JSON file', f, file_name=modified_file_path):
                    st.success("File downloaded successfully!")
                    st.session_state.file_processed = False
                    #st.experimental_rerun() 

        else:
            st.error("Uploaded data is not a JSON array.")

    else:
        # 创建输入框
        # 定义或获取 session_state 中的变量，用于保存输入框内容
        if 'question_body' not in st.session_state:
            st.session_state['question_body'] = ""
        if 'answer1_body' not in st.session_state:
            st.session_state['answer1_body'] = ""
        if 'answer2_body' not in st.session_state:
            st.session_state['answer2_body'] = ""
        if 'reference' not in st.session_state:
            st.session_state['reference'] = ""

        # 创建输入框，使用 session_state 变量作为值
        question_body = st.text_input("Question:", value=st.session_state['question_body'], key='question_body')
        answer1_body = st.text_input("Answer 1:", value=st.session_state['answer1_body'], key='answer1_body')
        answer2_body = st.text_input("Answer 2:", value=st.session_state['answer2_body'], key='answer2_body')
        reference = st.text_input("Reference:", value=st.session_state['reference'], key='reference')

        # 提交按钮，用于填充模板并显示结果
        if st.button("Submit"):
            # 根据reference输入是否为空选择不同的模板文件
            if reference:
                template_file = "./prompt_template/PAIRWISE_WR.txt"
            else:
                template_file = "./prompt_template/PAIRWISE_WOR.txt"
            
            # 尝试从文件加载基本prompt模板
            try:
                with open(template_file, "r") as file:
                    base_prompt = file.read()
            except FileNotFoundError:
                st.error(f"Error: {template_file} file not found.")
                st.stop()
            
            # 使用读取的模板和用户输入生成最终的prompt
            # Format the prompt based on input and template
            if not selected_criteria :
                final_prompt = base_prompt.format(
                    scenario = st.session_state.selected_option,
                    criteria = read_criteria(st.session_state.selected_option),
                    question_body=question_body,
                    answer1_body=answer1_body,
                    answer2_body=answer2_body,
                    reference=reference if reference else "N/A"  # Use "N/A" or any suitable placeholder if reference is empty
                )
            else:
                final_prompt = base_prompt.format(
                    scenario = st.session_state.selected_option,
                    criteria = user_selected_criteria(selected_criteria),
                    question_body=question_body,
                    answer1_body=answer1_body,
                    answer2_body=answer2_body,
                    reference=reference if reference else "N/A"  # Use "N/A" or any suitable placeholder if reference is empty
                )

            for_chart = {"reference":reference, "answer1":answer1_body, "answer2":answer2_body}

            # Call ollama.chat to process the dialogue
            #with st.chat_message("assistant"):
            message_placeholder = st.empty()
            results_holder = st.empty()

            full_response = ""

            with st.spinner('Runing...Please Wait...'):
                start_output = False
                # 获取模型响应的完整内容
                for chunk in ollama.chat(
                    model=st.session_state["model_name"],
                    messages=[{"role": "user", "content": final_prompt}],
                    stream=True
                ):
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += (chunk['message']['content'] or "")
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)

                        # 一旦收到第一个内容块，结束加载提示（通过结束 `with` 自动消失）
                        if not start_output:
                            start_output = True
            message_placeholder.empty()

            # 经过处理后再输出
            processed_response = extract_gpt_response_info_pairwise(full_response)

            # 结果显示
            final_result = str(processed_response["final_results"]).replace("Final Result: ", "")
            result_text = "🤝 It's a Tie!" if final_result == "Tie" else f"🏆 {final_result} Wins!"

            # 使用Streamlit的columns进行布局优化
            col1, col2 = st.columns([1, 3])

            # 更新后的样式
            common_style = "padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
            background_color1 = "background-color: #f8f9fa;"  # 浅灰色
            background_color2 = "background-color: #e9ecef;"  # 较深灰色

            with col1:
                st.markdown(f"""
                    <div style="{background_color1} {common_style}">
                        <h2 style="color: #007BFF;">{result_text}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div style="{background_color2} {common_style}">
                        <h3 style="color: #6c757d;">Detailed Evaluation Feedback</h3>
                        <p style="font-size: 16px; line-height: 1.6;">
                            {processed_response["Detailed_Evaluation_Feedback"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            # # 添加分隔符
            # st.markdown("<hr>", unsafe_allow_html=True)
            # 添加分隔符，设置样式减少上下间距
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)  # 自定义上下边距

            # 画分数的对比条形图
            st.markdown("### Score Comparison Breakdown 📊")
            fig1, fig2, fig3 = plot_scores_PAIRWISE(processed_response, for_chart)

            # 利用三列来显示图表
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h5 style='margin: 0;'>Score Comparison for A vs B</h5>", unsafe_allow_html=True)  # 使用h5标签来减小字体
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("<h5 style='margin: 0;'>Comparison of Answer 1 with Reference</h5>", unsafe_allow_html=True)  # 使用h5标签来减小字体
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                st.markdown("<h5 style='margin: 0;'>Comparison of Answer 2 with Reference</h5>", unsafe_allow_html=True)  # 使用h5标签来减小字体
                st.plotly_chart(fig3, use_container_width=True)

else:

    # 创建输入框
    question_body = st.text_input("Question:")
    answer_body = st.text_input("Answer:")
    reference = st.text_input("Reference:")

    # 提交按钮，用于填充模板并显示结果
    if st.button("Submit"):
        # 根据reference输入是否为空选择不同的模板文件
        if reference:
            template_file = "./prompt_template/POINTWISE_WR.txt"
        else:
            template_file = "./prompt_template/POINTWISE_WOR.txt"
        
        # 尝试从文件加载基本prompt模板
        try:
            with open(template_file, "r") as file:
                base_prompt = file.read()
        except FileNotFoundError:
            st.error(f"Error: {template_file} file not found.")
            st.stop()
        
        # 使用读取的模板和用户输入生成最终的prompt
        # Format the prompt based on input and template
        if not selected_criteria:
            final_prompt = base_prompt.format(
                scenario = st.session_state.selected_option,
                criteria = read_criteria(st.session_state.selected_option),
                question_body=question_body,
                answer_body=answer_body,
                reference=reference if reference else "N/A"  # Use "N/A" or any suitable placeholder if reference is empty
            )
        else:
            final_prompt = base_prompt.format(
                scenario = st.session_state.selected_option,
                criteria = user_selected_criteria(selected_criteria),
                question_body=question_body,
                answer_body=answer_body,
                reference=reference if reference else "N/A"  # Use "N/A" or any suitable placeholder if reference is empty
            )

        for_chart = {"reference":reference, "answer":answer_body}

        score_placeholder = st.empty()
        message_placeholder = st.empty() 
        full_response = ""

        # 显示加载提示（正在执行）
        with st.spinner('Runing...Please Wait...'):
            start_output = False

            # 获取模型响应的完整内容
            for chunk in ollama.chat(
                model=st.session_state["model_name"],
                messages=[{"role": "user", "content": final_prompt}],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += (chunk['message']['content'] or "")
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)

                    # 一旦收到第一个内容块，结束加载提示（通过结束 `with` 自动消失）
                    if not start_output:
                        start_output = True
        message_placeholder.empty()
        # 处理响应
        processed_response = extract_gpt_response_info_pointwise(full_response)

        # 结果显示
        print(processed_response)
        overall_score = processed_response["Overall_Score"]["Overall Score"]
        result_text = f'📝 Final Score: <span style="color: #FF4500;">{overall_score}/10</span></h2>'

        # 使用Streamlit的columns进行布局优化
        col1, col2 = st.columns([1, 3])

        # 更新后的样式
        common_style = "padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
        background_color1 = "background-color: #f8f9fa;"  # 浅灰色
        background_color2 = "background-color: #e9ecef;"  # 较深灰色

        with col1:
            st.markdown(f"""
                <div style="{background_color1} {common_style}">
                    <h2 style="color: #007BFF;">{result_text}</h2>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="{background_color2} {common_style}">
                    <h3 style="color: #6c757d;">Detailed Evaluation Feedback</h3>
                    <p style="font-size: 16px; line-height: 1.6;">
                        {processed_response["Detailed_Evaluation_Feedback"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # 添加分隔符，设置样式减少上下间距
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)  # 自定义上下边距

        # 画分数的对比条形图
        st.markdown("### Score Comparison Breakdown 📊")
        fig1, fig2= plot_scores_POINTWISE(processed_response, for_chart)

        # 利用三列来显示图表
        col1, col2= st.columns(2)
        with col1:
            st.markdown("<h5 style='margin: 0;'>Score Comparison for A vs B</h5>", unsafe_allow_html=True)  # 使用h5标签来减小字体
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("<h5 style='margin: 0;'>Comparison of Answer with Reference</h5>", unsafe_allow_html=True)  # 使用h5标签来减小字体
            st.plotly_chart(fig2, use_container_width=True)