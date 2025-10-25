#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型配置管理模块
支持本地 Ollama 和硅基流动 API
"""

import subprocess
import requests
import os

def get_ollama_models():
    """获取本地 Ollama 模型列表"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
            return models
        return []
    except Exception as e:
        print(f"获取 Ollama 模型失败: {e}")
        return []

def call_ollama(model, prompt, stream=False):
    """调用本地 Ollama 模型"""
    import ollama
    if stream:
        return ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
    else:
        return ollama.generate(model=model, prompt=prompt)['response']

def call_siliconflow(model, prompt, api_key, stream=False):
    """
    调用硅基流动 API
    文档: https://docs.siliconflow.cn/
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    try:
        if stream:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
            response.raise_for_status()
            return response
        else:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"硅基流动 API 调用失败: {str(e)}")

# 硅基流动支持的热门模型
SILICONFLOW_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "01-ai/Yi-Lightning",
]

def get_available_models():
    """获取所有可用模型（本地 + API）"""
    models = {
        'local': get_ollama_models(),
        'siliconflow': SILICONFLOW_MODELS
    }
    return models

def call_model(model_name, prompt, model_type='local', api_key=None, stream=False):
    """
    统一模型调用接口
    
    Args:
        model_name: 模型名称
        prompt: 提示词
        model_type: 'local' 或 'siliconflow'
        api_key: API密钥（仅API调用需要）
        stream: 是否流式输出
    """
    if model_type == 'local':
        return call_ollama(model_name, prompt, stream)
    elif model_type == 'siliconflow':
        if not api_key:
            raise ValueError("使用硅基流动 API 需要提供 API Key")
        return call_siliconflow(model_name, prompt, api_key, stream)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
