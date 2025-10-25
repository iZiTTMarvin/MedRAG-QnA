#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG医疗问答系统 - 无Neo4j启动版本
当Neo4j数据库不可用时，系统仍可演示NER和大模型问答功能
"""

import streamlit as st
import sys
import os

def show_startup_info():
    """显示启动信息"""
    st.markdown("""
    # 🏥 RAG医疗问答系统
    
    ## 📋 当前系统状态
    
    ### ✅ 已加载模块：
    - 🤖 **BERT实体识别模型** - 支持8类医疗实体识别
    - 🧠 **Qwen2.5大语言模型** - 支持意图识别和答案生成
    - 📊 **数据增强策略** - 实体替换、掩码、拼接技术
    - 🔍 **TF-IDF实体对齐** - 提升实体识别准确性
    
    ### ⚠️ Neo4j状态：
    - 📊 **知识图谱功能受限** - Neo4j数据库未连接
    - 💡 **建议操作**：启动Neo4j服务获得完整RAG功能
    
    ## 🚀 可用功能：
    
    1. **实体识别演示** - 输入医疗问题，查看NER结果
    2. **意图识别演示** - 体验16类医疗意图分类
    3. **大模型问答** - 基于训练数据的医疗咨询回答
    4. **系统架构展示** - 了解RAG技术实现原理
    
    ---
    
    ### 🔧 完整功能启用方法：
    
    1. **启动Neo4j数据库**：
       ```bash
       # 设置用户名: neo4j, 密码: password
       # 确保端口7474可访问
       ```
    
    2. **构建知识图谱**（首次使用）：
       ```bash
       python build_up_graph.py --website http://localhost:7474 --user neo4j --password password --dbname neo4j
       ```
    
    3. **重新启动系统**：
       ```bash
       streamlit run login.py
       ```
    
    ---
    
    **⭐ 开始体验简化版功能，或按上述步骤启用完整RAG功能！**
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="RAG医疗问答系统",
        page_icon="🏥",
        layout="wide"
    )
    
    show_startup_info()
    
    # 提供登录入口
    if st.button("🚀 进入系统（简化版）", type="primary"):
        # 导入主系统
        try:
            from login import main as login_main
            # 设置session state以跳过登录
            st.session_state.logged_in = True
            st.session_state.admin = True
            st.session_state.usname = "演示用户"
            st.rerun()
        except Exception as e:
            st.error(f"系统启动失败: {e}")
    
    # 如果已登录，显示主界面
    if st.session_state.get('logged_in', False):
        try:
            from webui import main
            main(st.session_state.get('admin', False), st.session_state.get('usname', ''))
        except Exception as e:
            st.error(f"主系统加载失败: {e}")
            st.info("请检查依赖包是否完整安装")