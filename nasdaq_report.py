#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import requests
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# ========== 配置部分 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 监控清单：增加VIX、美元指数、半导体ETF、美债、原油
WATCHLIST = {
    "^IXIC": "纳斯达克指数",
    "^GSPC": "标普500指数",
    "^SOX": "费城半导体指数",
    "SMH": "半导体ETF",
    "MU": "美光科技",
    "WDC": "闪迪",
    "AMD": "AMD",
    "INTC": "英特尔",
    "NVDA": "英伟达",
    "^TNX": "10年期美债收益率",
    "CL=F": "WTI原油价格",
    "^VIX": "CBOE波动率指数(VIX)",
    "DX-Y.NYB": "美元指数"
}

# ========== 辅助函数：技术指标 ==========
def calculate_rsi(series: List[float], period: int = 14) -> float:
    """计算RSI（相对强弱指数）"""
    if len(series) < period + 1:
        return 50.0  # 默认中性
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_advanced_data(ticker: str, name: str) -> Dict[str, Any]:
    """
    获取增强行情数据：价格、涨跌幅、5日均价偏离、RSI、成交量比
    """
    try:
        stock = yf.Ticker(ticker)
        # 获取最近30个交易日数据，用于计算指标
        hist = stock.history(period="1mo")
        if len(hist) < 2:
            return {"name": name, "ticker": ticker, "price": "N/A", "change_pct": "N/A",
                    "price_vs_ma5": "N/A", "rsi": "N/A", "volume_ratio": "N/A"}
        
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
        change_pct = (current - prev) / prev * 100
        
        # 5日均价
        ma5 = hist['Close'].iloc[-5:].mean() if len(hist) >= 5 else current
        price_vs_ma5 = (current - ma5) / ma5 * 100
        
        # RSI (14日)
        closes = hist['Close'].tolist()
        rsi = calculate_rsi(closes, 14)
        
        # 成交量比：今日成交量 / 20日均量
        vol_20_avg = hist['Volume'].iloc[-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
        volume_ratio = hist['Volume'].iloc[-1] / vol_20_avg if vol_20_avg > 0 else 1.0
        
        return {
            "name": name,
            "ticker": ticker,
            "price": round(current, 2),
            "change_pct": round(change_pct, 2),
            "price_vs_ma5": round(price_vs_ma5, 2),
            "rsi": round(rsi, 1),
            "volume_ratio": round(volume_ratio, 2)
        }
    except Exception as e:
        print(f"获取 {ticker} 失败: {e}")
        return {"name": name, "ticker": ticker, "price": "Error", "change_pct": "Error",
                "price_vs_ma5": "Error", "rsi": "Error", "volume_ratio": "Error"}

def fetch_financial_data() -> Dict[str, Any]:
    """批量获取所有监控标的数据"""
    results = {}
    for ticker, name in WATCHLIST.items():
        results[name] = fetch_advanced_data(ticker, name)
    return results

# ========== 回调预警评分系统 ==========
def calculate_downturn_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于六维度指标计算回调风险分数 (0-100)
    分数越高，回调风险越大
    """
    score = 0
    warnings = []
    
    # 1. 技术面 (权重30)
    # 纳指相对5日均线偏离
    ndx = data.get("纳斯达克指数", {})
    if isinstance(ndx.get("price_vs_ma5"), (int, float)):
        vs_ma5 = ndx["price_vs_ma5"]
        if vs_ma5 > 5:
            score += 15
            warnings.append("纳指远高于5日均线(>5%)，短期超买")
        elif vs_ma5 > 2:
            score += 8
    
    # 半导体RSI
    sox = data.get("费城半导体指数", {})
    if isinstance(sox.get("rsi"), (int, float)):
        rsi = sox["rsi"]
        if rsi > 80:
            score += 20
            warnings.append("费半RSI > 80，极端超买")
        elif rsi > 70:
            score += 10
    
    # 广度: 可用纳指/标普涨跌幅差异简单模拟（标普涨幅落后纳指 = 广度恶化）
    spx = data.get("标普500指数", {})
    if isinstance(ndx.get("change_pct"), (int, float)) and isinstance(spx.get("change_pct"), (int, float)):
        diff = ndx["change_pct"] - spx["change_pct"]
        if diff > 1.5:
            score += 10
            warnings.append("纳指大幅跑赢标普，市场广度恶化")
    
    # 2. 能源与通胀 (权重20)
    oil = data.get("WTI原油价格", {})
    if isinstance(oil.get("price"), (int, float)):
        if oil["price"] > 100:
            score += 20
            warnings.append(f"WTI原油 ${oil['price']} > 100，通胀压力剧烈")
        elif oil["price"] > 90:
            score += 12
    
    # 3. 情绪面 (权重15)
    vix = data.get("CBOE波动率指数(VIX)", {})
    if isinstance(vix.get("price"), (int, float)):
        if vix["price"] > 30:
            score += 15
            warnings.append(f"VIX > 30，恐慌情绪爆发")
        elif vix["price"] > 20:
            score += 8
    
    # 4. 利率环境 (权重15)
    tnx = data.get("10年期美债收益率", {})
    if isinstance(tnx.get("price"), (int, float)):
        if tnx["price"] > 5.0:
            score += 15
            warnings.append("10年期美债收益率 > 5%，对成长股估值构成压力")
        elif tnx["price"] > 4.7:
            score += 8
    
    # 5. 资金面简化指标：半导体成交量比 > 2 表示资金过度集中
    smh = data.get("半导体ETF", {})
    if isinstance(smh.get("volume_ratio"), (int, float)):
        if smh["volume_ratio"] > 2.0:
            score += 10
            warnings.append("半导体ETF成交量比 > 2，资金过度拥挤")
    
    # 6. 集中度风险 (权重10)
    mu = data.get("美光科技", {})
    wdc = data.get("闪迪", {})
    if isinstance(mu.get("change_pct"), (int, float)) and isinstance(wdc.get("change_pct"), (int, float)):
        if mu["change_pct"] > 10 or wdc["change_pct"] > 10:
            score += 8
            warnings.append("存储芯片龙头单日暴涨超10%，泡沫化情绪")
    
    # 总分封顶100
    score = min(score, 100)
    risk_level = "低风险" if score < 30 else "中风险" if score < 60 else "高风险" if score < 80 else "极高风险"
    
    return {
        "score": score,
        "risk_level": risk_level,
        "warnings": warnings if warnings else ["未检测到显著回调信号"]
    }

# ========== DeepSeek AI 分析 ==========
def call_deepseek(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        return "请配置 DEEPSEEK_API_KEY 环境变量"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
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
            return f"API 调用失败，状态码：{response.status_code}"
    except Exception as e:
        return f"请求异常：{str(e)}"

def generate_ai_analysis(data: Dict[str, Any], risk: Dict[str, Any]) -> str:
    """
    生成专业市场分析简报（六维度 + 预警评分）
    """
    # 格式化市场数据（包含技术指标）
    data_lines = []
    for name, info in data.items():
        if info.get("price") in ["N/A", "Error"]:
            continue
        price = info["price"]
        change = info["change_pct"]
        vs_ma5 = info.get("price_vs_ma5", "N/A")
        rsi = info.get("rsi", "N/A")
        vol_ratio = info.get("volume_ratio", "N/A")
        line = f"- **{name}**：{price}，涨跌幅 {change:+.2f}% | 偏离5日均: {vs_ma5:+.2f}% | RSI: {rsi} | 成交量比: {vol_ratio}"
        data_lines.append(line)
    data_text = "\n".join(data_lines)
    
    # 预警分数
    score_text = f"回调风险评分: {risk['score']}/100 ({risk['risk_level']})\n预警明细: " + "; ".join(risk['warnings'])
    
    prompt = f"""
你是资深美股策略师。基于以下数据和预警评分，撰写一份专业的纳斯达克回调风险日报。

数据日期: {datetime.now().strftime('%Y-%m-%d')}

【市场与个股数据（含技术指标）】
{data_text}

【回调预警评分系统】
{score_text}

请严格按照以下六维度框架输出分析报告：

### 一、基本面裂缝
- 结合就业、消费信心、通胀预期（参考已知宏观背景），判断当前经济是否出现滞胀信号。
- 给出看多 vs 看空的宏观基本面对比。

### 二、资金面动向
- 根据成交量比、机构持仓风格的简化判断（如半导体拥挤度），推断当前资金是否在撤离核心科技股。
- 如果内部人减持数据缺失，请基于现有数据给出谨慎推论。

### 三、技术面预警
- 分析纳指、费城半导体指数的RSI及偏离5日均线程度，判断是否超买。
- 指出是否存在“指数上涨、广度恶化”的背离。

### 四、情绪面与避险信号
- 结合VIX、Put/Call（若缺失可跳过），以及成交量异常放大迹象，描述市场情绪状态（麻木/恐慌/贪婪）。

### 五、地缘与能源压力
- 基于WTI原油价格及其对通胀预期的传导，评估潜在利空。

### 六、制度与监管风险
- 简要提及美债收益率曲线、美联储政策预期（如降息延迟、加息讨论）对科技股估值的压制。

### 七、综合结论与操作建议
- 给出今日整体风险评估（低/中/高/极高风险）。
- 列出最需要关注的3个指标（例如：VIX是否突破20，油价是否突破100，纳指能否守住26000）。
- 最后输出一句核心警告。

要求：语气专业冷峻，直接给出判断，避免模糊词汇。
"""
    return call_deepseek(prompt)

# ========== 推送模块 ==========
def send_to_dingtalk(webhook_url: str, content: str) -> bool:
    headers = {"Content-Type": "application/json"}
    truncated = content[:4000] + "..." if len(content) > 4000 else content
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "📊 纳指监控日报", "text": truncated}
    }
    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"钉钉发送失败：{e}")
        return False

def send_to_feishu(webhook_url: str, secret: str, content: str) -> bool:
    import hashlib, hmac, base64, time
    timestamp = str(int(time.time()))
    sign_str = f"{timestamp}\n{secret}"
    sign = base64.b64encode(hmac.new(secret.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    headers = {"Content-Type": "application/json"}
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": "📊 纳指监控日报",
                    "content": [[{"tag": "text", "text": content[:3000]}]]
                }
            }
        }
    }
    full_url = f"{webhook_url}?timestamp={timestamp}&sign={sign}"
    try:
        resp = requests.post(full_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200 and resp.json().get("code") == 0
    except Exception as e:
        print(f"飞书发送失败：{e}")
        return False

# ========== 主函数 ==========
def main():
    print(f"开始生成简报：{datetime.now()}")
    
    # 1. 获取增强数据
    print("正在获取行情及技术指标...")
    data = fetch_financial_data()
    if not data:
        print("数据获取失败")
        return
    
    # 2. 计算回调风险评分
    print("计算回调风险评分...")
    risk = calculate_downturn_score(data)
    
    # 3. AI 深度分析
    print("正在调用 AI 分析...")
    analysis = generate_ai_analysis(data, risk)
    
    # 4. 格式化报告（Markdown）
    today = datetime.now().strftime("%Y-%m-%d")
    report = f"# 📊 纳指监控日报\n\n**日期**：{today}\n\n"
    report += "## 📈 市场数据与技术指标\n\n"
    for name, info in data.items():
        if info.get("price") in ["N/A", "Error"]:
            continue
        emoji = "🟢" if isinstance(info["change_pct"], (int, float)) and info["change_pct"] >= 0 else "🔴"
        report += f"- {emoji} **{name}**：{info['price']} ({info['change_pct']:+.2f}%)  |  RSI:{info['rsi']}  |  量比:{info['volume_ratio']}\n"
    
    report += f"\n## ⚠️ 回调风险评分\n**{risk['score']}/100 - {risk['risk_level']}**\n\n"
    if risk['warnings']:
        report += "**预警信号**：\n" + "\n".join(f"- {w}" for w in risk['warnings']) + "\n"
    
    report += f"\n## 🤖 AI 六维分析\n\n{analysis}\n\n"
    report += "---\n*报告由 AI 生成，仅供参考，不构成任何投资建议。*"
    
    # 5. 推送（优先钉钉，可选飞书）
    dingtalk_url = os.environ.get("DINGTALK_WEBHOOK", "")
    feishu_url = os.environ.get("FEISHU_WEBHOOK", "")
    feishu_secret = os.environ.get("FEISHU_SECRET", "")
    
    if dingtalk_url:
        if send_to_dingtalk(dingtalk_url, report):
            print("钉钉推送成功")
        else:
            print("钉钉推送失败")
    
    if feishu_url and feishu_secret:
        if send_to_feishu(feishu_url, feishu_secret, report):
            print("飞书推送成功")
        else:
            print("飞书推送失败")
    
    print("简报生成完毕")

if __name__ == "__main__":
    main()
