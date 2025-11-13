import os
import streamlit as st
import ner_model as zwk
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re
import model_config



@st.cache_resource
def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #加载ChatGLM模型
    # glm_tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b-128k", trust_remote_code=True)
    # glm_model = AutoModel.from_pretrained("model/chatglm3-6b-128k",trust_remote_code=True,device=device)
    # glm_model.eval()
    glm_model = None
    glm_tokenizer= None
    #加载Bert模型
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()
    model_name = 'model/chinese-roberta-wwm-ext'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    # 加载训练好的权重文件
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', map_location=device))
    
    bert_model = bert_model.to(device)
    bert_model.eval()
    return glm_tokenizer,glm_model,bert_tokenizer,bert_model,idx2tag,rule,tfidf_r,device



def Intent_Recognition(query, model_name, model_type='local', api_key=None):
    # 针对简单常见问题，使用规则快速匹配
    simple_intents = {
        '怎么办': ['简介', '治疗', '药品', '检查'],
        '吃什么': ['药品', '宜吃'],
        '不能吃': ['忌吃'],
        '症状': ['简介', '症状'],
        '原因': ['简介', '病因'],
        '预防': ['简介', '预防'],
        '检查': ['简介', '检查'],
        '治疗': ['简介', '治疗', '药品'],
        '并发': ['简介', '并发'],
        '生产': ['生产商']
    }
    
    # 检查是否匹配简单规则
    for keyword, intents in simple_intents.items():
        if keyword in query:
            intent_list = []
            for intent in intents:
                if intent == '简介':
                    intent_list.append('查询疾病简介')
                elif intent == '治疗':
                    intent_list.append('查询疾病的治疗方法')
                elif intent == '药品':
                    intent_list.append('查询疾病所需药品')
                elif intent == '宜吃':
                    intent_list.append('查询疾病宜吃食物')
                elif intent == '忌吃':
                    intent_list.append('查询疾病忌吃食物')
                elif intent == '检查':
                    intent_list.append('查询疾病所需检查项目')
                elif intent == '症状':
                    intent_list.append('查询疾病的症状')
                elif intent == '病因':
                    intent_list.append('查询疾病病因')
                elif intent == '预防':
                    intent_list.append('查询疾病预防措施')
                elif intent == '并发':
                    intent_list.append('查询疾病的并发疾病')
                elif intent == '生产商':
                    intent_list.append('查询药品的生产商')
            result = str(intent_list) + f" # 根据关键词'{keyword}'匹配"
            print(f'意图识别结果(规则匹配):{result}')
            return result
    
    # 如果没有规则匹配，使用简化的LLM提示
    prompt = f"""
你是医疗意图识别专家。分析用户问题："{query}"

从以下类别选择最相关的（可多选，最多3个）：
- 查询疾病简介
- 查询疾病病因
- 查询疾病预防措施
- 查询疾病所需药品
- 查询疾病宜吃食物
- 查询疾病忌吃食物
- 查询疾病所需检查项目
- 查询疾病的症状
- 查询疾病的治疗方法
- 查询疾病的并发疾病
- 查询药品的生产商

直接输出：["类别1", "类别2"]
"""
    try:
        rec_result = model_config.call_model(model_name, prompt, model_type, api_key, stream=False)
        print(f'意图识别结果(LLM-{model_name}):{rec_result}')
        return rec_result
    except Exception as e:
        print(f'意图识别失败: {e}')
        return "[查询疾病简介] # 默认意图"


def add_shuxing_prompt(entity,shuxing,client):
    add_prompt = ""
    if client is None:
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{shuxing}需求，但Neo4j数据库未连接，无法查询知识图谱。"
        add_prompt += f"</提示>"
        return add_prompt
        
    try:
        sql_q = "match (a:疾病{名称:'%s'}) return a.%s" % (entity,shuxing)
        res_data = client.run(sql_q).data()
        if not res_data:
            warning_msg = f"知识图谱中未找到{entity}的{shuxing}信息。"
            st.warning(warning_msg)
            return f"<提示>用户对{entity}可能有查询{shuxing}需求，但知识库暂无信息。</提示>"

        res = res_data[0].values()
        add_prompt+=f"<提示>"
        add_prompt+=f"用户对{entity}可能有查询{shuxing}需求，知识库内容如下："
        if len(res)>0:
            join_res = "".join(res)
            add_prompt+=join_res
        else:
            add_prompt+="图谱中无信息，查找失败。"
            st.warning(f"知识图谱中{entity}的{shuxing}字段为空。")
        add_prompt+=f"</提示>"
    except Exception as e:
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{shuxing}需求，但查询知识图谱时发生错误：{str(e)[:30]}。"
        add_prompt += f"</提示>"
    return add_prompt
def add_lianxi_prompt(entity,lianxi,target,client):
    add_prompt = ""
    if client is None:
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{lianxi}需求，但Neo4j数据库未连接，无法查询知识图谱。"
        add_prompt += f"</提示>"
        return add_prompt
        
    try:
        sql_q = "match (a:疾病{名称:'%s'})-[r:%s]->(b:%s) return b.名称" % (entity,lianxi,target)
        res = client.run(sql_q).data()#[0].values()
        res = [list(data.values())[0] for data in res]
        add_prompt+=f"<提示>"
        add_prompt+=f"用户对{entity}可能有查询{lianxi}需求，知识库内容如下："
        if len(res)>0:
            join_res = "、".join(res)
            add_prompt+=join_res
        else:
            add_prompt+="图谱中无信息，查找失败。"
        add_prompt+=f"</提示>"
    except Exception as e:
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{lianxi}需求，但查询知识图谱时发生错误：{str(e)[:30]}。"
        add_prompt += f"</提示>"
    return add_prompt
def generate_prompt(response,query,client,bert_model, bert_tokenizer,rule, tfidf_r, device, idx2tag):
    entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    # print(response)
    # print(entities)
    yitu = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>"
    prompt +="<指令>请你仅针对医疗类问题提供简洁和专业的回答。如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”，以明确告知你的回答限制。</指令>"
    if '疾病症状' in entities and  '疾病' not in entities:
        if client is not None:
            try:
                sql_q = "match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:'%s'}) return a.名称" % (entities['疾病症状'])
                res_data = client.run(sql_q).data()
                if not res_data:
                    st.warning(f"知识图谱缺少症状[{entities['疾病症状']}]到疾病的关联数据。")
                    prompt+=f"<提示>用户有{entities['疾病症状']}的情况，但知识库缺少相关关联数据，无法推测相关疾病。</提示>"
                else:
                    res = list(res_data[0].values())
                    # print('res=',res)
                    if len(res)>0:
                        entities['疾病'] = random.choice(res)
                        all_en = "、".join(res)
                        prompt+=f"<提示>用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。请注意这只是一个推测，你需要明确告知用户这一点。</提示>"
                    else:
                        st.warning(f"症状[{entities['疾病症状']}]关联疾病字段为空。")
                        prompt+=f"<提示>用户有{entities['疾病症状']}的情况，但知识库缺少相关关联数据，无法推测相关疾病。</提示>"
            except Exception as e:
                prompt+=f"<提示>用户有{entities['疾病症状']}的情况，但查询知识图谱时发生错误，无法推测相关疾病。</提示>"
        else:
            prompt+=f"<提示>用户有{entities['疾病症状']}的情况，但Neo4j数据库未连接，无法查询相关疾病信息。</提示>"
    pre_len = len(prompt)
    if "简介" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病简介',client)
            yitu.append('查询疾病简介')
    if "病因" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病病因',client)
            yitu.append('查询疾病病因')
    if "预防" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'预防措施',client)
            yitu.append('查询预防措施')
    if "治疗周期" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'治疗周期',client)
            yitu.append('查询治疗周期')
    if "治愈概率" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'治愈概率',client)
            yitu.append('查询治愈概率')
    if "易感人群" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病易感人群',client)
            yitu.append('查询疾病易感人群')
    if "药品" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病使用药品','药品',client)
            yitu.append('查询疾病使用药品')
    if "宜吃食物" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病宜吃食物','食物',client)
            yitu.append('查询疾病宜吃食物')
    if "忌吃食物" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病忌吃食物','食物',client)
            yitu.append('查询疾病忌吃食物')
    if "检查项目" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病所需检查','检查项目',client)
            yitu.append('查询疾病所需检查')
    if "查询疾病所属科目" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病所属科目','科目',client)
            yitu.append('查询疾病所属科目')
    # if "所属科目" in response:
    #     if '疾病' in entities:
    #         prompt+=add_lianxi_prompt(entities['疾病'],'疾病所属科目','科目')
    #         yitu.append('查询疾病所属科目')
    if "症状" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病的症状','疾病症状',client)
            yitu.append('查询疾病的症状')
    if "治疗" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'治疗的方法','治疗方法',client)
            yitu.append('查询治疗的方法')
    if "并发" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病并发疾病','疾病',client)
            yitu.append('查询疾病并发疾病')
    if "生产商" in response:
        if client is not None and '药品' in entities:
            try:
                sql_q = "match (a:药品商)-[r:生产]->(b:药品{名称:'%s'}) return a.名称" % (entities['药品'])
                res_data = client.run(sql_q).data()
                if not res_data:
                    st.warning(f"知识图谱缺少药品[{entities['药品']}]的生产商关联数据。")
                    prompt+=f"<提示>用户对{entities['药品']}可能有查询药品生产商的需求，但知识库缺少相关数据。</提示>"
                else:
                    res = res_data[0].values()
                    prompt+=f"<提示>"
                    prompt+=f"用户对{entities['药品']}可能有查询药品生产商的需求，知识图谱内容如下："
                    if len(res)>0:
                        prompt+="".join(res)
                    else:
                        prompt+="图谱中无信息，查找失败"
                        st.warning(f"药品[{entities['药品']}]的生产商字段为空。")
                    prompt+=f"</提示>"
            except Exception as e:
                prompt+=f"<提示>查询药品生产商时发生错误：{str(e)[:30]}</提示>"
        else:
            if '药品' in entities:
                prompt+=f"<提示>Neo4j数据库未连接，无法查询{entities['药品']}的生产商信息。</提示>"
            else:
                prompt+=f"<提示>未识别到药品实体，无法查询生产商信息。</提示>"
        yitu.append('查询药物生产商')
    if pre_len==len(prompt) :
        # 如果没有找到相关信息，但是用户的问题可能是问候或一般性咨询        
        if any(word in query.lower() for word in ['你好', 'hello', 'hi', '介绍', '帮助', '什么']):
            prompt += f"<提示>用户可能是在问候或询问系统功能。请介绍你是一个专业的医疗RAG问答系统，可以回答医疗相关问题，包括疾病简介、症状、治疗方法、药物信息等。请鼓励用户提出具体的医疗问题。</提示>"
        else:
            prompt += f"<提示>提示：知识库异常，没有相关信息！请你直接回答“根据已知信息无法回答该问题”！</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>现在你已经知道给定的“<提示></提示>”和“<用户问题></用户问题>”了,你要极其认真的判断提示里是否有用户问题所需的信息，如果没有相关信息，你必须直接回答“根据已知信息无法回答该问题”。</注意>"

    prompt += f"<注意>你一定要再次检查你的回答是否完全基于“<提示></提示>”的内容，不可产生提示之外的答案！换而言之，你的任务是根据用户的问题，将“<提示></提示>”整理成有条理、有逻辑的语句。你起到的作用仅仅是整合提示的功能，你一定不可以利用自身已经存在的知识进行回答，你必须从提示中找到问题的答案！</注意>"
    prompt += f"<注意>你必须充分的利用提示中的知识，不可将提示中的任何信息遗漏，你必须做到对提示信息的充分整合。你回答的任何一句话必须在提示中有所体现！如果根据提示无法给出答案，你必须回答“根据已知信息无法回答该问题”。<注意>"
    
    
    print(f'prompt:{prompt}')
    return prompt,"、".join(yitu),entities



# def ans_stream(prompt):
    
#     result = ""
#     for res,his in glm_model.stream_chat(glm_tokenizer, prompt, history=[]):
#         yield res



def main(is_admin, usname):
    cache_model = 'best_roberta_rnn_model_ent_aug'
    st.title(f"医疗智能问答机器人")

    with st.sidebar:
        col1, col2 = st.columns([0.6, 0.6])
        with col1:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                image_path = os.path.join(current_dir, "img", "logo.jpg")
                st.image(image_path, use_column_width=True)
            except Exception as e:
                st.error(f"无法加载图片: {str(e)}")

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button('新建对话窗口'):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('请选择对话窗口:', window_options)
        active_window_index = int(selected_window.split()[1]) - 1

        # 动态模型选择
        st.markdown("---")
        st.subheader("🤖 模型配置")
        
        # 获取可用模型
        available_models = model_config.get_available_models()
        
        # 模型来源选择
        model_source = st.radio(
            "模型来源",
            options=['💻 本地 Ollama', '☁️ 硅基流动 API'],
            horizontal=True
        )
        
        if model_source == '💻 本地 Ollama':
            model_type = 'local'
            api_key = None
            if available_models['local']:
                # 默认选中 deepseek-r1:8b
                default_idx = 0
                if 'deepseek-r1:8b' in available_models['local']:
                    default_idx = available_models['local'].index('deepseek-r1:8b')
                
                choice = st.selectbox(
                    '选择本地模型:',
                    options=available_models['local'],
                    index=default_idx
                )
            else:
                st.warning('⚠️ 未检测到本地 Ollama 模型')
                st.info('请运行: ollama pull deepseek-r1:8b')
                choice = 'deepseek-r1:8b'  # 默认
        else:
            model_type = 'siliconflow'
            api_key = st.text_input(
                '🔑 硅基流动 API Key',
                type='password',
                help='在 https://cloud.siliconflow.cn/ 获取 API Key'
            )
            choice = st.selectbox(
                '选择 API 模型:',
                options=model_config.SILICONFLOW_MODELS,
                index=1  # 默认 DeepSeek-R1
            )

        show_ent = show_int = show_prompt = False
        if is_admin:
            show_ent = st.sidebar.checkbox("显示实体识别结果")
            show_int = st.sidebar.checkbox("显示意图识别结果")
            show_prompt = st.sidebar.checkbox("显示查询的知识库信息")
            if st.button('修改知识图谱'):
            # 显示一个链接，用户可以点击这个链接在新标签页中打开百度
                st.markdown('[点击这里修改知识图谱](http://127.0.0.1:7474/)', unsafe_allow_html=True)



        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.experimental_rerun()

    glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
    
    # 延迟连接Neo4j，只在需要时连接
    client = None
    neo4j_connected = False
    
    # 在侧边栏添加Neo4j密码配置（仅管理员可见）
    custom_password = None
    if is_admin:
        with st.sidebar.expander("🔧 Neo4j 配置", expanded=False):
            custom_password = st.text_input(
                "Neo4j 密码（选填）",
                type="password",
                help="如果默认密码连接失败，请输入你的 Neo4j 密码"
            )
    
    # 尝试多种连接方式
    connection_attempts = [
        {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'neo4j'},
        {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'password'},
        {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'asd2528836683'},
        {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': '12345678'},
        {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'admin'},
        {'uri': 'http://localhost:7474', 'user': 'neo4j', 'password': 'neo4j'},
        {'uri': 'http://localhost:7474', 'user': 'neo4j', 'password': 'password'}
    ]
    
    # 如果管理员输入了自定义密码，优先尝试
    if custom_password:
        connection_attempts.insert(0, {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': custom_password})
    
    for attempt in connection_attempts:
        try:
            if attempt['uri'].startswith('bolt'):
                client = py2neo.Graph(attempt['uri'], auth=(attempt['user'], attempt['password']))
            else:
                client = py2neo.Graph(attempt['uri'], user=attempt['user'], password=attempt['password'], name='neo4j')
            
            # 测试连接
            client.run("RETURN 1")
            st.sidebar.success(f"✅ Neo4j数据库连接成功 ({attempt['uri']})")
            neo4j_connected = True
            break
        except Exception as e:
            continue
    
    if not neo4j_connected:
        st.sidebar.error("❌ Neo4j数据库连接失败")
        if is_admin:
            st.sidebar.info("💡 请在上方'Neo4j 配置'中输入正确的密码")
        else:
            st.sidebar.info("💡 提示：请联系管理员检查 Neo4j 配置")
        st.sidebar.info("💡 默认尝试密码: neo4j, password, asd2528836683")
        client = None

    current_messages = st.session_state.messages[active_window_index]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if show_ent:
                    with st.expander("实体识别结果"):
                        st.write(message.get("ent", ""))
                if show_int:
                    with st.expander("意图识别结果"):
                        st.write(message.get("yitu", ""))
                if show_prompt:
                    with st.expander("点击显示知识库信息"):
                        st.write(message.get("prompt", ""))

    if query := st.chat_input("Ask me anything!", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        response_placeholder = st.empty()
        response_placeholder.text("正在进行意图识别...")

        query = current_messages[-1]["content"]
        response = Intent_Recognition(query, choice, model_type, api_key)
        response_placeholder.empty()

        prompt, yitu, entities = generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)

        last = ""
        try:
            if model_type == 'local':
                for chunk in ollama.chat(model=choice, messages=[{'role': 'user', 'content': prompt}], stream=True):
                    last += chunk['message']['content']
                    response_placeholder.markdown(last)
            else:  # siliconflow
                if not api_key:
                    last = "⚠️ 请在侧边栏输入硅基流动 API Key"
                else:
                    import json
                    stream_response = model_config.call_siliconflow(choice, prompt, api_key, stream=True)
                    for line in stream_response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data_str = line[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            last += content
                                            response_placeholder.markdown(last)
                                except:
                                    continue
        except Exception as e:
            last = f"❌ 生成答案失败: {str(e)}"
        response_placeholder.markdown(last)

        knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
        zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
        with st.chat_message("assistant"):
            st.markdown(last)
            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(str(entities))
            if show_int:
                with st.expander("意图识别结果"):
                    st.write(yitu)
            if show_prompt:
                
                
                with st.expander("点击显示知识库信息"):
                    st.write(zhishiku_content)
        current_messages.append({"role": "assistant", "content": last, "yitu": yitu, "prompt": zhishiku_content, "ent": str(entities)})


    st.session_state.messages[active_window_index] = current_messages
