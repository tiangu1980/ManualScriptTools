import pandas as pd
import re
from collections import defaultdict

def categorize_complaint(text):
    """
    根据抱怨内容进行分类
    """
    if pd.isna(text) or text.strip() == "" or text.lower() == "no":
        return "Empty/No complaint"
    
    text_lower = text.lower().strip()
    
    # 纯情绪表达，没有实质内容
    emotion_patterns = [
        r"^i\s*(don'?t|do\s*not)\s*(like|want)\s*(it|this|that)\.?$",
        r"^(annoying|bothers?\s*me|irritating)\.?!*$",
        r"^(not\s*interested|no\s*interest)\.?$",
        r"^i'?m\s*not\s*interested\.?$",
        r"^don'?t\s*want\s*(it|this|that)\.?$",
        r"^i\s*don'?t\s*want\s*(it|this|that)\s*anymore\.?$",
        r"^murx\.?$",
        r"^unnecessary\.?$"
    ]
    
    for pattern in emotion_patterns:
        if re.match(pattern, text_lower):
            return "I don't like it."
    
    # 性能和资源问题
    if any(keyword in text_lower for keyword in ['ram', 'memory', 'slow', 'battery', 'power', 'drain', 'performance', 'system']):
        if 'ram' in text_lower or 'memory' in text_lower:
            return "Memory/RAM usage issues"
        elif 'battery' in text_lower or 'power' in text_lower or 'drain' in text_lower:
            return "Battery/Power consumption issues"
        elif 'slow' in text_lower or 'performance' in text_lower:
            return "System performance issues"
        else:
            return "System resource issues"
    
    # 存储空间问题
    if any(keyword in text_lower for keyword in ['space', 'storage', 'disk']):
        return "Storage space issues"
    
    # 功能问题和Bug
    if any(keyword in text_lower for keyword in ['bug', 'error', 'problem', 'freeze', 'crash', 'update', 'installation']):
        if 'freeze' in text_lower or 'frozen' in text_lower:
            return "App freezing issues"
        elif 'update' in text_lower:
            return "Update related problems"
        elif 'installation' in text_lower or 'install' in text_lower:
            return "Installation issues"
        else:
            return "Technical bugs/errors"
    
    # 界面和用户体验问题
    if any(keyword in text_lower for keyword in ['background', 'wallpaper', 'image', 'change', 'desktop']):
        if 'change' in text_lower and ('background' in text_lower or 'wallpaper' in text_lower):
            return "Unwanted wallpaper/background changes"
        elif "can't change" in text_lower:
            return "Cannot change wallpaper settings"
        else:
            return "Wallpaper/Image related issues"
    
    # Bing相关问题
    if 'bing' in text_lower:
        return "Bing search engine complaints"
    
    # 自动启用问题
    if any(keyword in text_lower for keyword in ['re-enabling', 'keeps', 'automatically', 'constant']):
        return "Auto-enabling/Persistent behavior"
    
    # 隐私和数据问题
    if any(keyword in text_lower for keyword in ['data', 'privacy', 'home', 'hacked', 'account']):
        return "Privacy/Data concerns"
    
    # 意外安装
    if any(keyword in text_lower for keyword in ['mistake', 'downloaded by mistake', 'promotion']):
        return "Accidental installation"
    
    # 政策相关
    if 'policy' in text_lower:
        return "Company/Internal policy"
    
    # 期望不符
    if any(keyword in text_lower for keyword in ['expectation', 'doesn\'t meet']):
        return "Doesn't meet expectations"
    
    # 分心问题
    if any(keyword in text_lower for keyword in ['distract', 'focus', 'school', 'work']):
        return "Distraction concerns"
    
    # 其他具体功能抱怨
    if len(text_lower) > 20 and any(keyword in text_lower for keyword in ['because', 'when', 'but', 'however']):
        return "Specific functionality complaints"
    
    # 默认归类为一般不满
    return "I don't like it."

def analyze_feedback():
    """
    分析反馈数据
    """
    # 读取CSV文件
    df = pd.read_csv("d:/GrowthAI/2 Feedback/2 UnInstallFeedback_filtered.csv")
    
    # 创建分类列
    df['Complaint_Category'] = df['Uninstallreason'].apply(categorize_complaint)
    
    # 按照要求的维度进行分组统计
    result = df.groupby(['AppVersion', 'Market', 'Language', 'Country/Region', 'Complaint_Category']).size().reset_index(name='Count')
    
    # 重新排列列的顺序
    result = result[['AppVersion', 'Market', 'Language', 'Country/Region', 'Complaint_Category', 'Count']]
    
    # 显示分类汇总
    category_summary = df['Complaint_Category'].value_counts()
    print("抱怨分类汇总:")
    print("=" * 50)
    for category, count in category_summary.items():
        print(f"{category}: {count} 条")
    
    print(f"\n总计: {len(df)} 条反馈")
    
    # 保存详细结果
    result.to_csv("d:/GrowthAI/2 Feedback/feedback_analysis_detailed.csv", index=False, encoding='utf-8-sig')
    
    # 保存分类汇总
    category_summary.to_csv("d:/GrowthAI/2 Feedback/feedback_categories_summary.csv", header=['Count'], encoding='utf-8-sig')
    
    print(f"\n详细分析结果已保存到: feedback_analysis_detailed.csv")
    print(f"分类汇总已保存到: feedback_categories_summary.csv")
    
    return result, category_summary

if __name__ == "__main__":
    result, summary = analyze_feedback()
    
    # 显示前20行详细结果作为示例
    print("\n详细分析结果示例 (前20行):")
    print("=" * 80)
    print(result.head(20).to_string(index=False))
