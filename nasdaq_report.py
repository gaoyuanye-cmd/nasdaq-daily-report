#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import requests
import yfinance as yf
from datetime import datetime
from typing import Dict, Any

# ========== 配置部分 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 需要监控的标的（代码:显示名称）
WATCHLIST = {
    "^IXIC": "纳斯达克指数",
    "^GSPC": "标普500指数",
    "^SOX": "费城半导体指数",
    "MU": "美光科技",
    "WDC": "闪迪",
    "AMD": "AMD",
    "INTC": "英特尔",
    "NVDA": "英伟达",
    "^TNX": "10年期美债收益率",
    "CL=F": "WTI原油价格"
}

# ========== 数据获取 ==========
def fetch_financial_data() -> Dict[str, Any]:
    """从 Yahoo Finance 获取行情数据"""
    results = {}
    for ticker, name in WATCHLIST.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change_pct = (current - prev) / prev * 100
                results[name] = {
                    "price": round(current, 2),
                    "change_pct": round(change_pct, 2)
                }
            else:
                results[name] = {"price": "N/A", "change_pct": "N/A"}
        except Exception as e:
            print(f"获取 {ticker} 失败: {e}")
            results[name] = {"price": "Error", "change_pct": "Error"}
    return results

# ========== DeepSeek 分析 ==========
def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API 进行市场分析"""
    if not DEEPSEEK_API_KEY:
        return "请配置 DEEPSEEK_API_KEY 环境变量"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业的华尔街分析师，擅长解读市场数据并撰写每日简报。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"API 调用失败：{response.status_code}"
    except Exception as e:
        return f"请求异常：{str(e)}"

def generate_ai_analysis(data: Dict[str, Any]) -> str:
    """生成市场分析简报"""
    # 格式化数据
    data_lines = []
    for name, info in data.items():
        price = info['price']
        change_val = info['change_pct']
        if isinstance(change_val, (int, float)):
            change_str = f"{change_val:+.2f}%"
        else:
            change_str = str(change_val)
        data_lines.append(f"- {name}：{price}，涨跌幅 {change_str}")
    data_text = "\n".join(data_lines)

    prompt = f"""
以下是今日美股收盘数据（美东时间 {datetime.now().strftime('%Y-%m-%d')}）：

{data_text}

请按以下格式撰写简报：

一、大盘综述（2-3 句话总结当日市场走势，判断看多/看空/中性）
二、芯片与 AI 板块焦点（点评存储芯片龙头表现，是否存在集中度风险）
三、多空简评（用一句话给出核心观点）

要求：简洁专业，直接切中数据要点。
"""
    return call_deepseek(prompt)

# ========== 推送模块 ==========
def send_to_wework(webhook_url: str, content: str) -> bool:
    """发送 Markdown 格式消息到企业微信群"""
    headers = {"Content-Type": "application/json"}
    truncated = content[:4000] + "..." if len(content) > 4000 else content
    payload = {
        "msgtype": "markdown",
        "markdown": {"content": truncated}
    }
    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200 and resp.json().get("errcode") == 0
    except Exception as e:
        print(f"企业微信发送失败：{e}")
        return False

def send_to_dingtalk(webhook_url: str, content: str) -> bool:
    """发送消息到钉钉群"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": "纳指监控日报",
            "text": content
        }
    }
    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"钉钉发送失败：{e}")
        return False

# ========== 主函数 ==========
def main():
    print(f"开始生成简报：{datetime.now()}")
    
    # 1. 获取数据
    print("正在获取行情数据...")
    data = fetch_financial_data()
    if not data:
        print("数据获取失败")
        return
    
    # 2. AI 分析
    print("正在调用 AI 分析...")
    analysis = generate_ai_analysis(data)
    
    # 3. 格式化报告
    today = datetime.now().strftime("%Y-%m-%d")
    report = f"# 📊 纳指监控日报\n\n**日期**：{today}\n\n"
    report += "## 📈 市场数据\n\n"
    for name, info in data.items():
        change_val = info['change_pct']
        if isinstance(change_val, (int, float)):
            change_str = f"+{change_val}%" if change_val >= 0 else f"{change_val}%"
        else:
            change_str = str(change_val)
        emoji = "🟢" if (isinstance(change_val, (int, float)) and change_val >= 0) else "🔴"
        report += f"- {emoji} **{name}**：{info['price']}（{change_str}）\n"
    report += f"\n## 🤖 AI 分析\n\n{analysis}\n\n"
    report += "---\n*报告由 AI 生成，仅供参考，不构成任何投资建议。*"
    
    # 4. 推送
    wework_url = os.environ.get("WEWORK_WEBHOOK", "")
    dingtalk_url = os.environ.get("DINGTALK_WEBHOOK", "")
    
    if wework_url:
        if send_to_wework(wework_url, report):
            print("企业微信推送成功")
        else:
            print("企业微信推送失败")
    
    if dingtalk_url:
        if send_to_dingtalk(dingtalk_url, report):
            print("钉钉推送成功")
        else:
            print("钉钉推送失败")
    
    print("简报生成完毕")

if __name__ == "__main__":
    main()
