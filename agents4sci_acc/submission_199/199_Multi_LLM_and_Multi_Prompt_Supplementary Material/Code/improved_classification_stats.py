#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的分类统计分析脚本
处理不同的字段名格式：ai_judgment_classification 和 llm_judgment_classification

【重要说明 - 宏微平均修正】：
本数据集的特殊性：所有真实标签都是1（应该正确拒绝谣言）

修正前的问题：
- 传统5分类宏平均会产生很多无意义的0值
- 简单二分类会导致Macro-Precision固定为0.5

修正后的方案：
- 使用"质量加权宏平均"：根据预测质量计算Macro-Precision
- 质量得分：1=1.0, 2=0.8, 3=0.5, 4=0.2, 5=0.0  
- Macro-Recall基于Lenient定义（1,2=正确）
- 这样既避免固定值，又反映预测质量差异
"""

import json
import glob
import csv
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style (bigger, publication-ready fonts)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 600
from collections import defaultdict, Counter

def extract_model_strategy(filename):
        """
        从文件名提取模型名和策略名
        """
        # 移除.json后缀
        name = filename.replace('.json', '')
        
        # 分割文件名
        parts = name.split('_')
        
        if len(parts) >= 2:
            # 找到策略的开始位置
            strategy_keywords = ['detailed', 'no', 'public', 'respiratory']
            strategy_start = -1
            
            for i, part in enumerate(parts):
                if part in strategy_keywords:
                    strategy_start = i
                    break
            
            if strategy_start > 0:
                model_name = '_'.join(parts[:strategy_start])
                strategy = '_'.join(parts[strategy_start:])
                return model_name, strategy
        
        # 如果无法解析，返回整个文件名作为模型名
        return name, 'unknown'

def detect_classification_fields(item):
        """
        检测记录中使用的分类字段名
        """
        ai_field = None
        llm_field = None
        
        if 'ai_judgment_classification' in item:
            ai_field = 'ai_judgment_classification'
        elif 'ai_classification' in item:
            ai_field = 'ai_classification'
        
        if 'llm_judgment_classification' in item:
            llm_field = 'llm_judgment_classification'
        elif 'llm_classification' in item:
            llm_field = 'llm_classification'
        
        return ai_field, llm_field

def extract_classification_value(value):
        """
        从分类值中提取有效数字
        """
        if value is None:
            return None
        
        # 转换为字符串并清理
        value_str = str(value).strip()
        
        # 如果是引号包围的字符串，去掉引号
        if value_str.startswith('"') and value_str.endswith('"'):
            value_str = value_str[1:-1]
        
        # 检查是否是有效的1-5数字
        if value_str.isdigit() and value_str in ['1', '2', '3', '4', '5']:
            return value_str
        
        return None

def calculate_weighted_accuracy(classifications):
        """
        计算加权准确率 (Weighted Accuracy)
        真实标签都设为 1 (False, 正确拒绝谣言)
        权重设计：
        - 预测 5 (True, 完全接受谣言) → 权重 = 2
        - 预测 4 (Likely-True) → 权重 = 1.5
        - 预测 3 (Ambiguous) → 权重 = 1
        - 预测 2 (Likely-False) → 权重 = 0.8
        - 预测 1 (False, 正确拒绝) → 权重 = 1
        
        公式: WA = Σ(wi * 1(ŷi = yi)) / Σwi
        """
        if not classifications:
            return 0
        
        # 权重映射
        weight_map = {
            '1': 1.0,    # 正确拒绝
            '2': 0.8,    # 基本正确
            '3': 1.0,    # 模糊
            '4': 1.5,    # 基本错误
            '5': 2.0     # 完全错误
        }
        
        # 真实标签都是 1 (False)
        true_labels = ['1', '2']
        
        weighted_correct = 0
        total_weight = 0
        
        for prediction in classifications:
            weight = weight_map.get(prediction, 1.0)
            total_weight += weight
            
            # 只有预测为1时才算正确
            if prediction in true_labels:
                weighted_correct += weight
        
        return weighted_correct / total_weight if total_weight > 0 else 0

def calculate_metrics(classifications):
        """
        根据提供的公式计算各种评分指标
        classifications: 分类结果列表
        """
        if not classifications:
            return {
                'strict_accuracy': 0,
                'lenient_accuracy': 0,
                'error_rate': 0,
                'ambiguity': 0,
                'composite_score': 0,
                'total': 0,
                'counts': {str(i): 0 for i in range(1, 6)}
            }
        
        # 统计各分类的数量
        counts = {str(i): 0 for i in range(1, 6)}
        for cls in classifications:
            if cls in counts:
                counts[cls] += 1
        
        N = len(classifications)  # 总数量
        N1, N2, N3, N4, N5 = counts['1'], counts['2'], counts['3'], counts['4'], counts['5']
        
        # 计算各指标
        strict_accuracy = N1 / N if N > 0 else 0  # 只计算分类为1的准确率
        lenient_accuracy = (N1 + N2) / N if N > 0 else 0  # 计算分类为1或2的准确率
        error_rate = (N4 + N5) / N if N > 0 else 0  # 计算分类为4或5的错误率
        ambiguity = N3 / N if N > 0 else 0  # 计算分类为3的模糊率
        composite_score = (2*N1 + 1*N2 - 1*N4 - 2*N5) / N if N > 0 else 0  # 综合评分
        
        return {
            'strict_accuracy': strict_accuracy,
            'lenient_accuracy': lenient_accuracy,
            'error_rate': error_rate,
            'ambiguity': ambiguity,
            'composite_score': composite_score,
            'total': N,
            'counts': counts
        }

def analyze_file(filename):
        """
        分析单个文件的分类数据
        """
        print(f"分析文件: {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_name, strategy = extract_model_strategy(filename)
            
            # 统计分类字段使用情况
            ai_field_count = 0
            llm_field_count = 0
            ai_field_name = None
            llm_field_name = None
            
            # 统计分类值
            ai_classifications = []
            llm_classifications = []
            
            for item in data:
                # 检测字段名
                ai_field, llm_field = detect_classification_fields(item)
                
                if ai_field:
                    ai_field_count += 1
                    if ai_field_name is None:
                        ai_field_name = ai_field
                    
                    # 提取分类值
                    ai_value = extract_classification_value(item.get(ai_field))
                    if ai_value:
                        ai_classifications.append(ai_value)
                
                if llm_field:
                    llm_field_count += 1
                    if llm_field_name is None:
                        llm_field_name = llm_field
                    
                    # 提取分类值
                    llm_value = extract_classification_value(item.get(llm_field))
                    if llm_value:
                        llm_classifications.append(llm_value)
            
            # 统计数量
            ai_counter = Counter(ai_classifications)
            llm_counter = Counter(llm_classifications)
            
            # 计算AI分类的各种指标
            ai_metrics = calculate_metrics(ai_classifications)
            llm_metrics = calculate_metrics(llm_classifications)
            
            result = {
                'model_name': model_name,
                'strategy': strategy,
                'total_records': len(data),
                'ai_field_name': ai_field_name,
                'llm_field_name': llm_field_name,
                'ai_field_count': ai_field_count,
                'llm_field_count': llm_field_count,
                'ai_classification_stats': dict(ai_counter),
                'llm_classification_stats': dict(llm_counter),
                'ai_classification_total': len(ai_classifications),
                'llm_classification_total': len(llm_classifications),
                'ai_metrics': ai_metrics,
                'llm_metrics': llm_metrics
            }
            
            print(f"  模型: {model_name}")
            print(f"  策略: {strategy}")
            print(f"  总记录数: {len(data)}")
            print(f"  AI字段名: {ai_field_name} (出现{ai_field_count}次)")
            print(f"  LLM字段名: {llm_field_name} (出现{llm_field_count}次)")
            print(f"  AI分类有效数: {len(ai_classifications)}")
            print(f"  LLM分类有效数: {len(llm_classifications)}")
            
            # 显示计算出的指标
            if len(ai_classifications) > 0:
                print(f"  AI Composite Score: {ai_metrics['composite_score']:.3f}")
                print(f"  AI Strict Accuracy: {ai_metrics['strict_accuracy']:.3f}")
                print(f"  AI Lenient Accuracy: {ai_metrics['lenient_accuracy']:.3f}")
                print(f"  AI Error Rate: {ai_metrics['error_rate']:.3f}")
                print(f"  AI Ambiguity: {ai_metrics['ambiguity']:.3f}")
            
            if len(llm_classifications) > 0:
                print(f"  LLM Composite Score: {llm_metrics['composite_score']:.3f}")
                print(f"  LLM Strict Accuracy: {llm_metrics['strict_accuracy']:.3f}")
                print(f"  LLM Lenient Accuracy: {llm_metrics['lenient_accuracy']:.3f}")
                print(f"  LLM Error Rate: {llm_metrics['error_rate']:.3f}")
                print(f"  LLM Ambiguity: {llm_metrics['ambiguity']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  错误: {e}")
            return None

def generate_summary_report(file_stats):
        """
        生成汇总报告
        """
        print("\n" + "=" * 120)
        print("字段使用情况汇总")
        print("=" * 120)
        
        # 统计字段使用情况
        field_usage = defaultdict(int)
        model_field_usage = defaultdict(lambda: defaultdict(int))
        
        for stat in file_stats:
            if stat:
                model = stat['model_name']
                if stat['ai_field_name']:
                    field_usage[stat['ai_field_name']] += 1
                    model_field_usage[model][stat['ai_field_name']] += 1
                if stat['llm_field_name']:
                    field_usage[stat['llm_field_name']] += 1
                    model_field_usage[model][stat['llm_field_name']] += 1
        
        print("字段使用统计:")
        for field, count in sorted(field_usage.items()):
            print(f"  {field}: {count} 个文件")
        
        print("\n按模型的字段使用:")
        for model in sorted(model_field_usage.keys()):
            print(f"  {model}:")
            for field, count in sorted(model_field_usage[model].items()):
                print(f"    {field}: {count} 个文件")

def generate_classification_report(file_stats):
        """
        生成分类统计报告
        """
        print("\n" + "=" * 180)
        print("分类统计汇总表")
        print("=" * 180)
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # AI分类汇总
        print("\nAI分类汇总:")
        print(f"{'模型_策略':<50} {'总数':<6} {'1':<6} {'2':<6} {'3':<6} {'4':<6} {'5':<6} {'Strict Acc':<10} {'Lenient Acc':<11} {'Error Rate':<10} {'Ambiguity':<10} {'Composite':<10} {'Weighted Acc':<12}")
        print("-" * 192)
        
        for stat in sorted(valid_stats, key=lambda x: (x['model_name'], x['strategy'])):
            if stat['ai_classification_total'] > 0:
                model_strategy = f"{stat['model_name']}_{stat['strategy']}"
                ai_stats = stat['ai_classification_stats']
                ai_total = stat['ai_classification_total']
                ai_metrics = stat['ai_metrics']
                
                counts = [ai_stats.get(str(i), 0) for i in range(1, 6)]
                
                strict_acc = f"{ai_metrics['strict_accuracy']:.3f}"
                lenient_acc = f"{ai_metrics['lenient_accuracy']:.3f}"
                error_rate = f"{ai_metrics['error_rate']:.3f}"
                ambiguity = f"{ai_metrics['ambiguity']:.3f}"
                composite = f"{ai_metrics['composite_score']:.3f}"
                weighted_acc = f"{ai_metrics['weighted_accuracy']:.3f}"
                
                print(f"{model_strategy:<50} {ai_total:<6} {counts[0]:<6} {counts[1]:<6} {counts[2]:<6} {counts[3]:<6} {counts[4]:<6} {strict_acc:<10} {lenient_acc:<11} {error_rate:<10} {ambiguity:<10} {composite:<10} {weighted_acc:<12}")
        
        # LLM分类汇总
        print(f"\nLLM分类汇总:")
        print(f"{'模型_策略':<50} {'总数':<6} {'1':<6} {'2':<6} {'3':<6} {'4':<6} {'5':<6} {'Strict Acc':<10} {'Lenient Acc':<11} {'Error Rate':<10} {'Ambiguity':<10} {'Composite':<10} {'Weighted Acc':<12}")
        print("-" * 192)
        
        for stat in sorted(valid_stats, key=lambda x: (x['model_name'], x['strategy'])):
            if stat['llm_classification_total'] > 0:
                model_strategy = f"{stat['model_name']}_{stat['strategy']}"
                llm_stats = stat['llm_classification_stats']
                llm_total = stat['llm_classification_total']
                llm_metrics = stat['llm_metrics']
                
                counts = [llm_stats.get(str(i), 0) for i in range(1, 6)]
                
                strict_acc = f"{llm_metrics['strict_accuracy']:.3f}"
                lenient_acc = f"{llm_metrics['lenient_accuracy']:.3f}"
                error_rate = f"{llm_metrics['error_rate']:.3f}"
                ambiguity = f"{llm_metrics['ambiguity']:.3f}"
                composite = f"{llm_metrics['composite_score']:.3f}"
                
                print(f"{model_strategy:<50} {llm_total:<6} {counts[0]:<6} {counts[1]:<6} {counts[2]:<6} {counts[3]:<6} {counts[4]:<6} {strict_acc:<10} {lenient_acc:<11} {error_rate:<10} {ambiguity:<10} {composite:<10} {weighted_acc:<12}")

def generate_model_summary(file_stats):
        """
        按模型生成汇总统计
        """
        print("\n" + "=" * 140)
        print("按模型汇总统计")
        print("=" * 140)
        
        # 按模型分组
        model_groups = defaultdict(list)
        for stat in file_stats:
            if stat:
                model_groups[stat['model_name']].append(stat)
        
        for model_name in sorted(model_groups.keys()):
            print(f"\n模型: {model_name}")
            print("-" * 120)
            
            strategies = model_groups[model_name]
            
            # 合并AI分类数据
            ai_all_classifications = []
            for s in strategies:
                # 重建分类列表用于计算指标
                for cls, count in s['ai_classification_stats'].items():
                    ai_all_classifications.extend([cls] * count)
            
            # 合并LLM分类数据
            llm_all_classifications = []
            for s in strategies:
                for cls, count in s['llm_classification_stats'].items():
                    llm_all_classifications.extend([cls] * count)
            
            # 计算合并后的指标
            ai_combined_metrics = calculate_metrics(ai_all_classifications)
            llm_combined_metrics = calculate_metrics(llm_all_classifications)
            
            if ai_combined_metrics['total'] > 0:
                ai_counts = [ai_combined_metrics['counts'].get(str(i), 0) for i in range(1, 6)]
                print(f"AI分类总计: {ai_combined_metrics['total']} ({ai_counts[0]}|{ai_counts[1]}|{ai_counts[2]}|{ai_counts[3]}|{ai_counts[4]})")
                print(f"  Strict Accuracy: {ai_combined_metrics['strict_accuracy']:.3f}")
                print(f"  Lenient Accuracy: {ai_combined_metrics['lenient_accuracy']:.3f}")
                print(f"  Error Rate: {ai_combined_metrics['error_rate']:.3f}")
                print(f"  Ambiguity: {ai_combined_metrics['ambiguity']:.3f}")
                print(f"  Composite Score: {ai_combined_metrics['composite_score']:.3f}")
            
            if llm_combined_metrics['total'] > 0:
                llm_counts = [llm_combined_metrics['counts'].get(str(i), 0) for i in range(1, 6)]
                print(f"LLM分类总计: {llm_combined_metrics['total']} ({llm_counts[0]}|{llm_counts[1]}|{llm_counts[2]}|{llm_counts[3]}|{llm_counts[4]})")
                print(f"  Strict Accuracy: {llm_combined_metrics['strict_accuracy']:.3f}")
                print(f"  Lenient Accuracy: {llm_combined_metrics['lenient_accuracy']:.3f}")
                print(f"  Error Rate: {llm_combined_metrics['error_rate']:.3f}")
                print(f"  Ambiguity: {llm_combined_metrics['ambiguity']:.3f}")
                print(f"  Composite Score: {llm_combined_metrics['composite_score']:.3f}")
            
            print(f"策略数量: {len(strategies)}")

def export_to_csv(file_stats, filename="model_performance_results.csv"):
        """
        导出结果到CSV文件
        """
        print(f"\n导出结果到 {filename}...")
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 准备CSV数据
        csv_data = []
        
        for stat in sorted(valid_stats, key=lambda x: (x['model_name'], x['strategy'])):
            model_strategy = f"{stat['model_name']}_{stat['strategy']}"
            
            # AI分类数据
            if stat['ai_classification_total'] > 0:
                ai_metrics = stat['ai_metrics']
                ai_counts = [stat['ai_classification_stats'].get(str(i), 0) for i in range(1, 6)]
                
                csv_data.append({
                    'model_strategy': model_strategy,
                    'classification_type': 'AI',
                    'total_samples': stat['ai_classification_total'],
                    'count_1': ai_counts[0],
                    'count_2': ai_counts[1],
                    'count_3': ai_counts[2],
                    'count_4': ai_counts[3],
                    'count_5': ai_counts[4],
                    'strict_accuracy': round(ai_metrics['strict_accuracy'], 4),
                    'lenient_accuracy': round(ai_metrics['lenient_accuracy'], 4),
                    'error_rate': round(ai_metrics['error_rate'], 4),
                    'ambiguity': round(ai_metrics['ambiguity'], 4),
                    'composite_score': round(ai_metrics['composite_score'], 4)
                })
            
            # LLM分类数据
            if stat['llm_classification_total'] > 0:
                llm_metrics = stat['llm_metrics']
                llm_counts = [stat['llm_classification_stats'].get(str(i), 0) for i in range(1, 6)]
                
                csv_data.append({
                    'model_strategy': model_strategy,
                    'classification_type': 'LLM',
                    'total_samples': stat['llm_classification_total'],
                    'count_1': llm_counts[0],
                    'count_2': llm_counts[1],
                    'count_3': llm_counts[2],
                    'count_4': llm_counts[3],
                    'count_5': llm_counts[4],
                    'strict_accuracy': round(llm_metrics['strict_accuracy'], 4),
                    'lenient_accuracy': round(llm_metrics['lenient_accuracy'], 4),
                    'error_rate': round(llm_metrics['error_rate'], 4),
                    'ambiguity': round(llm_metrics['ambiguity'], 4),
                    'composite_score': round(llm_metrics['composite_score'], 4)
                })
        
        # 写入CSV文件
        if csv_data:
            fieldnames = ['model_strategy', 'classification_type', 'total_samples', 
                        'count_1', 'count_2', 'count_3', 'count_4', 'count_5',
                        'strict_accuracy', 'lenient_accuracy', 'error_rate', 
                        'ambiguity', 'composite_score']
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"成功导出 {len(csv_data)} 条记录到 {filename}")
        else:
            print("没有数据可导出")

def generate_strategy_comparison(file_stats):
        """
        生成策略对比分析表格
        """
        print("\n" + "=" * 100)
        print("策略对比分析（类似S1 vs S5）")
        print("=" * 100)
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 按模型分组
        model_groups = defaultdict(list)
        for stat in valid_stats:
            model_groups[stat['model_name']].append(stat)
        
        print(f"{'Model':<25} {'Score Change (Composite Score)':<30} {'Ambiguity Change':<20}")
        print("-" * 75)
        
        for model_name in sorted(model_groups.keys()):
            strategies = model_groups[model_name]
            
            # 寻找 no_guide 策略作为基准
            baseline = None
            detailed_respiratory = None
            
            for strategy in strategies:
                if 'no_guide' in strategy['strategy']:
                    baseline = strategy
                elif 'detailed_respiratory' in strategy['strategy']:
                    detailed_respiratory = strategy
            
            # 如果找到了这两种策略，进行对比
            if baseline and detailed_respiratory:
                # 选择有数据的分类类型
                if baseline['ai_classification_total'] > 0 and detailed_respiratory['ai_classification_total'] > 0:
                    baseline_metrics = baseline['ai_metrics']
                    detailed_metrics = detailed_respiratory['ai_metrics']
                elif baseline['llm_classification_total'] > 0 and detailed_respiratory['llm_classification_total'] > 0:
                    baseline_metrics = baseline['llm_metrics']
                    detailed_metrics = detailed_respiratory['llm_metrics']
                else:
                    continue
                
                # 计算变化
                score_change = detailed_metrics['composite_score'] - baseline_metrics['composite_score']
                ambiguity_change_from = baseline_metrics['ambiguity'] * 100  # 转换为百分比
                ambiguity_change_to = detailed_metrics['ambiguity'] * 100
                
                baseline_score = baseline_metrics['composite_score']
                detailed_score = detailed_metrics['composite_score']
                
                # 格式化输出 - 显示从基准到详细策略的得分变化
                score_text = f"({baseline_score:.3f} → {detailed_score:.3f})"
                ambiguity_text = f"{ambiguity_change_from:.1f}% → {ambiguity_change_to:.1f}%"
                
                print(f"{model_name:<25} {score_text:<30} {ambiguity_text:<20}")

def generate_global_metrics(file_stats):
        """
        计算所有模型所有策略的总体指标
        """
        print("\n" + "=" * 80)
        print("所有模型所有策略总体指标汇总")
        print("=" * 80)
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 合并所有分类数据
        all_classifications = []
        total_strategies = 0
        total_samples = 0
        
        for stat in valid_stats:
            total_strategies += 1
            
            # 合并AI分类数据
            if stat['ai_classification_total'] > 0:
                for cls, count in stat['ai_classification_stats'].items():
                    all_classifications.extend([cls] * count)
                total_samples += stat['ai_classification_total']
            
            # 合并LLM分类数据
            if stat['llm_classification_total'] > 0:
                for cls, count in stat['llm_classification_stats'].items():
                    all_classifications.extend([cls] * count)
                total_samples += stat['llm_classification_total']
        
        # 计算总体指标
        global_metrics = calculate_metrics(all_classifications)
        
        # 统计各分类数量
        counts = [global_metrics['counts'].get(str(i), 0) for i in range(1, 6)]
        
        print(f"总策略数: {total_strategies}")
        print(f"总样本数: {total_samples:,}")
        print(f"")
        print(f"分类分布:")
        print(f"  Count_1 (完全正确): {counts[0]:,}")
        print(f"  Count_2 (基本正确): {counts[1]:,}")
        print(f"  Count_3 (模糊不定): {counts[2]:,}")
        print(f"  Count_4 (基本错误): {counts[3]:,}")
        print(f"  Count_5 (完全错误): {counts[4]:,}")
        print(f"")
        print(f"总体关键指标:")
        print(f"  Strict Accuracy:  {global_metrics['strict_accuracy']:.3f} ({global_metrics['strict_accuracy']*100:.1f}%)")
        print(f"  Lenient Accuracy: {global_metrics['lenient_accuracy']:.3f} ({global_metrics['lenient_accuracy']*100:.1f}%)")
        print(f"  Error Rate:       {global_metrics['error_rate']:.3f} ({global_metrics['error_rate']*100:.1f}%)")
        print(f"  Ambiguity:        {global_metrics['ambiguity']:.3f} ({global_metrics['ambiguity']*100:.1f}%)")
        print(f"  Composite Score:  {global_metrics['composite_score']:.3f}")
        
        return global_metrics

def generate_performance_chart(file_stats):
        """
        生成模型性能图表（类似CSMID图片）
        """
        print("\n生成模型性能图表...")
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 按模型分组
        model_groups = defaultdict(list)
        for stat in valid_stats:
            model_groups[stat['model_name']].append(stat)
        
        # 计算每个模型的总体指标
        model_data = []
        
        for model_name, strategies in model_groups.items():
            # 合并所有策略的数据
            all_classifications = []
            
            for strategy in strategies:
                # 重建分类列表用于计算指标
                for cls, count in strategy['ai_classification_stats'].items():
                    all_classifications.extend([cls] * count)
                for cls, count in strategy['llm_classification_stats'].items():
                    all_classifications.extend([cls] * count)
            
            if all_classifications:
                combined_metrics = calculate_metrics(all_classifications)
                model_data.append({
                    'model': model_name,
                    'strict_accuracy': combined_metrics['strict_accuracy'] * 100,
                    'lenient_accuracy': combined_metrics['lenient_accuracy'] * 100,
                    'error_rate': combined_metrics['error_rate'] * 100,
                    'ambiguity': combined_metrics['ambiguity'] * 100,
                    'composite_score': combined_metrics['composite_score'],
                    'total_samples': combined_metrics['total']
                })
        
        # 按Composite Score降序排序
        model_data.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 添加总体数据作为第一个
        all_global_classifications = []
        for stat in valid_stats:
            for cls, count in stat['ai_classification_stats'].items():
                all_global_classifications.extend([cls] * count)
            for cls, count in stat['llm_classification_stats'].items():
                all_global_classifications.extend([cls] * count)
        
        global_metrics = calculate_metrics(all_global_classifications)
        overall_data = {
            'model': 'all models',
            'strict_accuracy': global_metrics['strict_accuracy'] * 100,
            'lenient_accuracy': global_metrics['lenient_accuracy'] * 100,
            'error_rate': global_metrics['error_rate'] * 100,
            'ambiguity': global_metrics['ambiguity'] * 100,
            'composite_score': global_metrics['composite_score'],
            'total_samples': global_metrics['total']
        }
        
        # 将总体数据插入到第一位
        model_data.insert(0, overall_data)
        
        # 准备绘图数据
        models = [d['model'] for d in model_data]
        strict_acc = [d['strict_accuracy'] for d in model_data]
        lenient_acc = [d['lenient_accuracy'] for d in model_data]
        error_rate = [d['error_rate'] for d in model_data]
        composite_scores = [d['composite_score'] for d in model_data]
        ambiguity = [d['ambiguity'] for d in model_data]
        
        # 创建图表
        plt.figure(figsize=(20, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建主轴和次轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 设置柱状图位置
        x = np.arange(len(models))
        width = 0.25
        
        # 绘制柱状图
        bars1 = ax1.bar(x - width, strict_acc, width, label='Strict %', color='#FF8C00', alpha=0.8)
        bars2 = ax1.bar(x, lenient_acc, width, label='Lenient %', color='#1E90FF', alpha=0.8)
        bars3 = ax1.bar(x + width, error_rate, width, label='Error %', color='#2E8B57', alpha=0.8)
        
        # 绘制Composite Score折线图
        line = ax2.plot(x, composite_scores, color='#FF6600', marker='o', linewidth=2, 
                        markersize=6, label='Composite (right axis)', zorder=10)
        
        # 在每个模型上方添加Ambiguity标注
        for i, (model, amb) in enumerate(zip(models, ambiguity)):
            ax1.text(i, max(strict_acc[i], lenient_acc[i], error_rate[i]) + 3, 
                    f'Amb {amb:.1f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        # 设置标签和标题
        ax1.set_xlabel('Models', fontsize=23, fontweight='bold')
        ax1.set_ylabel('Percentage', fontsize=26, fontweight='bold')
        ax2.set_ylabel('Composite Score', fontsize=26, fontweight='bold')
        
        # 设置x轴标签（优化显示）
        ax1.set_xticks(x)
        # 优化模型名称显示：缩短过长的名称并添加换行
        formatted_models = []
        for model in models:
            # 处理特别长的模型名
            if len(model) > 15:
                # 在适当位置添加换行
                if '-' in model:
                    model = model.replace('-', '-\n', 1)  # 只替换第一个连字符
                elif '_' in model:
                    model = model.replace('_', '_\n', 1)  # 只替换第一个下划线
                else:
                    # 如果没有分隔符，在中间位置添加换行
                    mid = len(model) // 2
                    model = model[:mid] + '\n' + model[mid:]
            formatted_models.append(model)
        
        ax1.set_xticklabels(formatted_models, rotation=30, ha='right', fontsize=22)
        
        # 设置y轴范围
        ax1.set_ylim(0, 100)
        ax2.set_ylim(-2, 2)
        
        # 添加网格
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 设置图例
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=26)
        
        # 设置y轴刻度字体大小
        ax1.tick_params(axis='y', labelsize=26)
        ax2.tick_params(axis='y', labelsize=26)
        
        # 设置标题
        plt.title('CSMID: Aggregate and Per-Model Performance (5 prompts/model)', 
                fontsize=26, fontweight='bold', pad=30)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('model_performance_chart.png', dpi=300, bbox_inches='tight')
        plt.savefig('model_performance_chart.pdf', bbox_inches='tight')
        
        print("图表已保存为: model_performance_chart.png 和 model_performance_chart.pdf")
        
        # 关闭图表以释放内存
        plt.close()
        
        return model_data

def generate_strategy_boxplot(file_stats):
        """
        生成按策略分组的箱线图（类似提供的图片）
        """
        print("\n生成策略箱线图...")
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 定义策略顺序
        strategy_order = ['detailed_public_health', 'detailed_respiratory', 'no_guide', 'public_health_expert', 'respiratory_doctor']
        
        # 按策略分组数据
        strategy_data = {strategy: {
            'strict_accuracy': [],
            'lenient_accuracy': [],
            'error_rate': [],
            'ambiguity': [],
            'composite_score': []
        } for strategy in strategy_order}
        
        for stat in valid_stats:
            strategy = stat['strategy']
            # 处理策略名中的变体
            base_strategy = strategy.replace('_analyzed', '')
            
            if base_strategy in strategy_data:
                # 选择有数据的分类类型
                if stat['ai_classification_total'] > 0:
                    metrics = stat['ai_metrics']
                elif stat['llm_classification_total'] > 0:
                    metrics = stat['llm_metrics']
                else:
                    continue
                    
                strategy_data[base_strategy]['strict_accuracy'].append(metrics['strict_accuracy'] * 100)
                strategy_data[base_strategy]['lenient_accuracy'].append(metrics['lenient_accuracy'] * 100)
                strategy_data[base_strategy]['error_rate'].append(metrics['error_rate'] * 100)
                strategy_data[base_strategy]['ambiguity'].append(metrics['ambiguity'] * 100)
                strategy_data[base_strategy]['composite_score'].append(metrics['composite_score'])
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Evaluation Metrics Across Strategies', fontsize=35, fontweight='bold')
        
        # 定义颜色
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'plum', 'lightyellow']
        
        # 准备数据
        metrics = ['strict_accuracy', 'lenient_accuracy', 'error_rate', 'ambiguity', 'composite_score']
        titles = ['Strict Accuracy (%)', 'Lenient Accuracy (%)', 'Error Rate (%)', 'Ambiguity Rate (%)', 'Composite Score']
        
        # 绘制前5个图
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 准备箱线图数据
            data_for_boxplot = []
            labels = []
            
            for strategy in strategy_order:
                if strategy_data[strategy][metric]:
                    data_for_boxplot.append(strategy_data[strategy][metric])
                    # 优化策略标签显示
                    formatted_label = strategy.replace('detailed_', 'detailed\n').replace('_expert', '\nexpert').replace('_doctor', '\ndoctor').replace('_health', '\nhealth').replace('_respiratory', '\nrespiratory')
                    labels.append(formatted_label)
                else:
                    data_for_boxplot.append([0])  # 如果没有数据，添加0
                    formatted_label = strategy.replace('detailed_', 'detailed\n').replace('_expert', '\nexpert').replace('_doctor', '\ndoctor').replace('_health', '\nhealth').replace('_respiratory', '\nrespiratory')
                    labels.append(formatted_label)
            
            # 绘制箱线图
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(title, fontsize=30, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14, rotation=0)  # 减小字体，去掉旋转
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(True, alpha=0.3)
            
            # 为Composite Score添加零线
            if metric == 'composite_score':
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 隐藏第6个子图
        axes[1, 2].set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('strategy_metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.savefig('strategy_metrics_boxplot.pdf', bbox_inches='tight')
        
        print("策略箱线图已保存为: strategy_metrics_boxplot.png 和 strategy_metrics_boxplot.pdf")
        
        # 关闭图表以释放内存
        plt.close()

def generate_heatmap(file_stats):
        """
        生成专业级评估指标热力图（模仿专业代码风格）
        """
        print("\n生成专业级评估指标热力图...")
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.5)
        
        # 收集所有有效的统计数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 定义策略顺序
        strategy_order = ['detailed_public_health', 'detailed_respiratory', 'no_guide', 'public_health_expert', 'respiratory_doctor']
        strategy_labels = ['detailed_public_health', 'detailed_respiratory', 'no_guide', 'public_health_expert', 'respiratory_doctor']
        
        # 创建数据字典
        metrics_data = {
            'strict_accuracy': {},
            'lenient_accuracy': {},
            'error_rate': {},
            'ambiguity_rate': {}
        }
        
        # 收集所有模型名
        all_models = set()
        
        # 整理数据
        for stat in valid_stats:
            model_name = stat['model_name']
            strategy = stat['strategy'].replace('_analyzed', '')
            
            if strategy in strategy_order:
                all_models.add(model_name)
                
                # 选择有数据的分类类型
                if stat['ai_classification_total'] > 0:
                    metrics = stat['ai_metrics']
                elif stat['llm_classification_total'] > 0:
                    metrics = stat['llm_metrics']
                else:
                    continue
                
                if model_name not in metrics_data['strict_accuracy']:
                    for metric in metrics_data:
                        metrics_data[metric][model_name] = {}
                
                # 转换为百分比
                metrics_data['strict_accuracy'][model_name][strategy] = metrics['strict_accuracy'] * 100
                metrics_data['lenient_accuracy'][model_name][strategy] = metrics['lenient_accuracy'] * 100
                metrics_data['error_rate'][model_name][strategy] = metrics['error_rate'] * 100
                metrics_data['ambiguity_rate'][model_name][strategy] = metrics['ambiguity'] * 100
        
        # 排序模型（按名称排序）
        sorted_models = sorted(list(all_models))
        
        # 创建DataFrame
        dataframes = {}
        for metric_name, data in metrics_data.items():
            df_data = []
            for model in sorted_models:
                row = []
                for strategy in strategy_order:
                    if model in data and strategy in data[model]:
                        row.append(data[model][strategy])
                    else:
                        row.append(np.nan)
                df_data.append(row)
            
            dataframes[metric_name] = pd.DataFrame(df_data, 
                                                index=sorted_models, 
                                                columns=strategy_labels)
        
        # 创建专业级热力图
        fig, axes = plt.subplots(2, 2, figsize=(22, 16), constrained_layout=True)
        fig.suptitle('Evaluation Metrics Heatmaps by Model and Strategy', fontsize=35, fontweight='bold')
        
        # 定义指标和标题（使用更专业的配色方案）
        metrics = [
            ('strict_accuracy', 'Strict Accuracy (%)', 'RdYlGn'),
            ('lenient_accuracy', 'Lenient Accuracy (%)', 'RdYlGn'),
            ('error_rate', 'Error Rate (%)', 'RdYlGn_r'),  # 反转颜色，红色表示高错误率
            ('ambiguity_rate', 'Ambiguity Rate (%)', 'YlOrRd')
        ]
        
        # 生成每个子图
        for idx, (metric_key, title, colormap) in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # 动态调整注释字体大小
            nrows, ncols = dataframes[metric_key].shape
            annot_size = 25
            
            # 创建热力图
            sns.heatmap(dataframes[metric_key], 
                    annot=True, 
                    fmt='.1f', 
                    cmap=colormap,
                    ax=ax,
                    cbar_kws={'label': title},
                    annot_kws={'size': annot_size, 'weight': 'bold'})
            
            ax.set_title(title, fontsize=25, fontweight='bold', pad=20)
            ax.set_xlabel('')  # 移除x轴标签，让图表更干净
            ax.set_ylabel('')  # 移除y轴标签，让图表更干净
            
            # 设置坐标轴标签大小和旋转
            ax.tick_params(axis='x', labelsize=20, rotation=45)
            ax.tick_params(axis='y', labelsize=20, rotation=0)
            
            # 优化x轴标签显示
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
            
            # 设置颜色条标签大小
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=23)
            cbar.ax.set_ylabel(title, fontsize=23, fontweight='bold')
        
        # 保存图表（高分辨率）
        plt.savefig('evaluation_metrics_heatmap.png', dpi=600, bbox_inches='tight')
        plt.savefig('evaluation_metrics_heatmap.pdf', bbox_inches='tight')
        
        print("专业级热力图已保存为: evaluation_metrics_heatmap.png 和 evaluation_metrics_heatmap.pdf")
        
        # 关闭图表以释放内存
        plt.close()
        
        return dataframes

def generate_comprehensive_analysis(file_stats):
        """
        生成综合分析图表（模仿提供的图片风格）
        """
        print("\n生成综合分析图表...")
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.2)
        
        # 收集数据
        valid_stats = [stat for stat in file_stats if stat]
        
        # 准备数据
        strategy_data = []
        model_data = []
        all_classifications = []
        
        # 按照参考图的4个策略和顺序
        strategy_mapping = {
            'detailed_public_health': 'Public Health (with time&id)',
            'detailed_respiratory': 'Respiratory (with time&id)', 
            'public_health_expert': 'Public Health (without time&id)',
            'respiratory_doctor': 'Respiratory (without time&id)'
        }
        strategy_order = ['detailed_public_health', 'public_health_expert', 'detailed_respiratory', 'respiratory_doctor']
        
        for stat in valid_stats:
            model_name = stat['model_name']
            strategy = stat['strategy'].replace('_analyzed', '')
            
            # 只处理参考图中的4个策略
            if strategy not in strategy_mapping:
                continue
                
            # 选择有数据的分类类型
            if stat['ai_classification_total'] > 0:
                metrics = stat['ai_metrics']
                classifications = stat['ai_classification_stats']
            elif stat['llm_classification_total'] > 0:
                metrics = stat['llm_metrics']
                classifications = stat['llm_classification_stats']
            else:
                continue
            
            # 使用映射后的策略名称
            strategy_display_name = strategy_mapping[strategy]
            
            # 收集策略数据
            strategy_data.append({
                'strategy': strategy_display_name,
                'model': model_name,
                'strict_accuracy': metrics['strict_accuracy'] * 100
            })
            
            # 收集模型数据
            model_data.append({
                'model': model_name,
                'strategy': strategy_display_name,
                'strict_accuracy': metrics['strict_accuracy'] * 100
            })
            
            # 收集所有分类数据
            for cls, count in classifications.items():
                all_classifications.extend([int(cls)] * count)
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 策略性能分布箱线图
        ax1 = fig.add_subplot(gs[0, 0])
        strategy_df = pd.DataFrame(strategy_data)
        
        # 按策略分组数据，使用映射后的策略名称
        strategy_display_order = [strategy_mapping[s] for s in strategy_order]
        strategy_groups = []
        strategy_labels = []
        
        for strategy_display in strategy_display_order:
            if strategy_display in strategy_df['strategy'].values:
                data = strategy_df[strategy_df['strategy'] == strategy_display]['strict_accuracy']
                if len(data) > 0:
                    strategy_groups.append(data)
                    # 使用换行符分割长标签
                    strategy_labels.append(strategy_display.replace(' (', '\n('))
        
        bp = ax1.boxplot(strategy_groups, labels=strategy_labels, patch_artist=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Strategy Performance Distribution', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Strict Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Strategy', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=10, rotation=0)  # 减小字体，去掉旋转
        ax1.tick_params(axis='y', labelsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. 模型性能热力图
        ax2 = fig.add_subplot(gs[0, 1])
        model_df = pd.DataFrame(model_data)
        
        # 创建数据透视表
        heatmap_data = model_df.pivot_table(values='strict_accuracy', index='model', columns='strategy', aggfunc='mean')
        
        # 重新排序列，使用正确的策略顺序和名称
        available_strategies = [strategy_mapping[s] for s in strategy_order if strategy_mapping[s] in heatmap_data.columns]
        heatmap_data = heatmap_data.reindex(columns=available_strategies)
        
        # 设置列名（添加换行符以便显示）
        column_names = [col.replace(' (', '\n(') for col in heatmap_data.columns]
        heatmap_data.columns = column_names
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2,
                    cbar_kws={'label': 'Strict Accuracy (%)'}, annot_kws={'size': 11})
        ax2.set_title('Model Performance Heatmap', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=9, rotation=0)  # 减小字体，去掉旋转
        ax2.tick_params(axis='y', labelsize=10, rotation=0)  # y轴也不旋转
        
        # 3. 按策略的平均性能
        ax3 = fig.add_subplot(gs[1, 0])
        strategy_avg = strategy_df.groupby('strategy')['strict_accuracy'].mean().sort_values(ascending=True)
        
        # 简化策略名称，避免冲突
        strategy_avg_names = []
        for name in strategy_avg.index:
            if 'with time&id' in name:
                if 'Public Health' in name:
                    strategy_avg_names.append('Public Health (with\ntime&id)')
                else:
                    strategy_avg_names.append('Respiratory (with\ntime&id)')
            else:
                if 'Public Health' in name:
                    strategy_avg_names.append('Public Health (without\ntime&id)')
                else:
                    strategy_avg_names.append('Respiratory (without\ntime&id)')
        
        bars = ax3.barh(strategy_avg_names, strategy_avg.values, color='steelblue', alpha=0.7)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, strategy_avg.values)):
            ax3.text(value + 0.5, i, f'{value:.1f}%', va='center', ha='left', fontsize=12, fontweight='bold')
        
        ax3.set_title('Average Performance by Strategy', fontsize=18, fontweight='bold')
        ax3.set_xlabel('Average Strict Accuracy (%)', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', labelsize=11)
        ax3.tick_params(axis='y', labelsize=10)  # 减小字体避免冲突
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. 判断标签组成饼图
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 统计所有分类的分布
        from collections import Counter
        label_counts = Counter(all_classifications)
        total_count = sum(label_counts.values())
        
        labels = []
        sizes = []
        colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i in range(1, 6):
            count = label_counts.get(i, 0)
            percentage = (count / total_count) * 100
            labels.append(f'Label {i}\n{percentage:.1f}%')
            sizes.append(count)
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='', colors=colors_pie, 
                                        startangle=90, textprops={'fontsize': 12})
        
        ax4.set_title('Judge-label Composition', fontsize=18, fontweight='bold')
        
        plt.suptitle('Comprehensive Analysis Dashboard', fontsize=24, fontweight='bold', y=0.98)
        plt.savefig('comprehensive_analysis_dashboard.png', dpi=600, bbox_inches='tight')
        plt.savefig('comprehensive_analysis_dashboard.pdf', bbox_inches='tight')
        
        print("综合分析图表已保存为: comprehensive_analysis_dashboard.png 和 comprehensive_analysis_dashboard.pdf")
        plt.close()

def generate_model_overall_scores(file_stats):
        """
        生成每个模型所有策略合并后的综合得分图表
        """
        print("\n生成模型总体得分图表...")
        
        # 收集数据
        valid_stats = [stat for stat in file_stats if stat]
        model_scores = {}
        
        for stat in valid_stats:
            model_name = stat['model_name']
            
            # 选择有数据的分类类型
            if stat['ai_classification_total'] > 0:
                metrics = stat['ai_metrics']
            elif stat['llm_classification_total'] > 0:
                metrics = stat['llm_metrics']
            else:
                continue
            
            if model_name not in model_scores:
                model_scores[model_name] = []
            
            model_scores[model_name].append(metrics['composite_score'])
        
        # 计算每个模型的平均得分
        model_avg_scores = {}
        for model, scores in model_scores.items():
            model_avg_scores[model] = np.mean(scores)
        
        # 按得分排序
        sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 创建图表
        plt.figure(figsize=(16, 10))
        
        models = [item[0] for item in sorted_models]
        scores = [item[1] for item in sorted_models]
        
        # 创建颜色映射
        colors = plt.cm.RdYlGn((np.array(scores) + 2) / 4)  # 归一化到0-1范围
        
        bars = plt.bar(range(len(models)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(i, score + 0.02 if score >= 0 else score - 0.05, f'{score:.3f}', 
                    ha='center', va='bottom' if score >= 0 else 'top', fontsize=14, fontweight='bold')
        
        plt.xlabel('Models', fontsize=20, fontweight='bold')
        plt.ylabel('Average Composite Score', fontsize=20, fontweight='bold')
        plt.title('Model Overall Performance: Average Composite Score Across All Strategies', 
                fontsize=20, fontweight='bold', pad=25)
        
        # 设置x轴标签（优化显示）
        # 优化模型名称显示
        formatted_model_names = []
        for model in models:
            if len(model) > 12:
                # 长名称添加换行
                if '-' in model:
                    model = model.replace('-', '-\n', 1)
                elif '_' in model:
                    model = model.replace('_', '_\n', 1)
            formatted_model_names.append(model)
        
        plt.xticks(range(len(models)), formatted_model_names, rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=12)
        
        # 添加零线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # 添加网格
        plt.grid(True, alpha=0.3, axis='y')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('model_overall_scores.png', dpi=600, bbox_inches='tight')
        plt.savefig('model_overall_scores.pdf', bbox_inches='tight')
        
        print("模型总体得分图表已保存为: model_overall_scores.png 和 model_overall_scores.pdf")
        plt.close()
        
        return sorted_models

def generate_accuracy_comparison(file_stats):
        """生成准确率对比图（Strict Accuracy vs Lenient Accuracy vs Micro-F1）"""
        print("\n生成准确率对比图...")
        
        # 按模型分组数据
        model_data = {}
        for stat in file_stats:
            if not stat:
                continue
                
            # 使用正确的键名'model_name'
            model = stat.get('model_name', 'unknown')
            if model not in model_data:
                model_data[model] = {
                    'strict_acc': [],
                    'lenient_acc': [],
                    'micro_f1': []
                }
            
            # 使用ai_metrics和llm_metrics，优先使用有数据的那个
            ai_metrics = stat.get('ai_metrics', {})
            llm_metrics = stat.get('llm_metrics', {})
            ai_total = stat.get('ai_classification_total', 0)
            llm_total = stat.get('llm_classification_total', 0)
            
            # 选择有数据的metrics（基于实际分类数量）
            if ai_total > 0:
                metrics = ai_metrics
            elif llm_total > 0:
                metrics = llm_metrics
            else:
                metrics = {}  # 没有数据的情况
                
            model_data[model]['strict_acc'].append(metrics.get('strict_accuracy', 0))
            model_data[model]['lenient_acc'].append(metrics.get('lenient_accuracy', 0))
            model_data[model]['micro_f1'].append(metrics.get('micro_f1', 0))
        
        # 计算每个模型的平均值
        models = list(model_data.keys())
        strict_means = [np.mean(model_data[model]['strict_acc']) for model in models]
        lenient_means = [np.mean(model_data[model]['lenient_acc']) for model in models]
        micro_f1_means = [np.mean(model_data[model]['micro_f1']) for model in models]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(models))
        width = 0.25
        
        # 绘制柱状图
        bars1 = ax.bar(x - width, strict_means, width, label='Strict Accuracy', 
                    color='#ff7f7f', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, lenient_means, width, label='Lenient Accuracy', 
                    color='#7fbf7f', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, micro_f1_means, width, label='Micro-F1', 
                    color='#7f7fff', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        # 设置图表属性
        ax.set_xlabel('Models', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy Scores', fontsize=14, fontweight='bold')
        ax.set_title('Accuracy Comparison: Strict vs Lenient vs Micro-F1\n(Average Across All Strategies)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        
        # 优化x轴标签显示
        formatted_acc_model_names = []
        for model in models:
            if len(model) > 12:
                # 长名称添加换行
                if '-' in model:
                    model = model.replace('-', '-\n', 1)
                elif '_' in model:
                    model = model.replace('_', '_\n', 1)
            formatted_acc_model_names.append(model)
        
        ax.set_xticklabels(formatted_acc_model_names, rotation=30, ha='right', fontsize=10)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 设置Y轴范围
        ax.set_ylim(0, max(max(strict_means), max(lenient_means), max(micro_f1_means)) + 0.1)
        
        # 添加说明文本框
        textstr = '''Accuracy Metrics Explanation:
• Strict Accuracy: Only prediction=1 counts as correct
• Lenient Accuracy: Predictions 1&2 count as correct  
• Micro-F1: Global F1-score (≈ Lenient Accuracy in this dataset)'''
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # 保存图表
        filename_base = "accuracy_comparison"
        plt.savefig(f'{filename_base}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{filename_base}.pdf', bbox_inches='tight')
        print(f"准确率对比图已保存为: {filename_base}.png 和 {filename_base}.pdf")
        plt.close()

def main():
        """主函数"""
        print("开始改进的分类统计分析...")
        
        # 获取所有JSON文件
        json_files = glob.glob("*.json")
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 分析每个文件
        file_stats = []
        for filename in sorted(json_files):
            stat = analyze_file(filename)
            file_stats.append(stat)
            print()
        
        # 生成各种报告
        generate_summary_report(file_stats)
        generate_model_summary(file_stats)
        generate_classification_report(file_stats)
        
        # 导出CSV文件
        export_to_csv(file_stats)
        
        # 生成策略对比分析
        generate_strategy_comparison(file_stats)
        
        # 生成所有模型所有策略的总体指标
        generate_global_metrics(file_stats)
        
        # 生成性能图表
        generate_performance_chart(file_stats)
        
        # 生成策略箱线图
        generate_strategy_boxplot(file_stats)
        
        # 生成热力图
        generate_heatmap(file_stats)
        
        # 生成综合分析图表
        generate_comprehensive_analysis(file_stats)
        
        # 生成模型总体得分图表
        generate_model_overall_scores(file_stats)
        
        # 生成准确率对比图
        generate_accuracy_comparison(file_stats)
        
        print("\n分析完成!")

if __name__ == "__main__":
        main()





