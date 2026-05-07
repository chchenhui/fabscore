import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot training vs non-training score distributions')
    parser.add_argument('--input-file', type=str, required=True, help='Path to scores.npz file')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    return parser.parse_args()

def plot_distributions(scores_data, output_dir):
    """Plot histogram distributions for training vs non-training scores"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # npzファイルの構造を確認
    # all_scores[model_id][dataset_name][method] = {"training": [...], "non-training": [...]}
    
    for model_id in scores_data.keys():
        model_scores = scores_data[model_id]
        
        for dataset_name in model_scores.keys():
            dataset_scores = model_scores[dataset_name]
            
            for method in dataset_scores.keys():
                method_scores = dataset_scores[method]
                
                # スコアデータを取得
                training_scores = method_scores['training']
                non_training_scores = method_scores["non-training"]
                
                # 新しい図を作成
                plt.figure(figsize=(10, 6))
                
                # Plot histograms
                plt.hist(training_scores, bins=30, alpha=0.7, label='Training', color='blue', density=True)
                plt.hist(non_training_scores, bins=30, alpha=0.7, label='Non-training', color='red', density=True)
                
                # add legend
                plt.legend(fontsize=14)
                
                # add title
                plt.title(f'{model_id} - {dataset_name} - {method}', fontsize=16)
                
                # add x label
                plt.xlabel('Score', fontsize=14)
                
                # add y label
                plt.ylabel('Density', fontsize=14)
                
                # save plot
                plot_filename = f'{model_id}_{dataset_name}_{method}_distribution.png'
                plot_filename = plot_filename.replace('/', '_')  # パス文字を置換
                plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
                print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
                
                # 図を閉じる（メモリリークを防ぐ）
                plt.close()
    
   
def main():
    args = parse_arguments()
    
    # Load scores data
    if not os.path.exists(args.input_file):
        print(f"Error: Scores file {args.input_file} not found")
        return
    
    print(f"Loading scores from {args.input_file}")
    
    # pickleファイルを読み込む
    with open(args.input_file, 'rb') as f:
        scores_data = pickle.load(f)
    
    # npzファイルの構造を表示
    print(f"Available keys: {list(scores_data.keys())}")
    
    # データ構造の詳細を表示
    for model_id in scores_data.keys():
        print(f"\nModel: {model_id}")
        model_scores = scores_data[model_id]
        model_scores = dict(model_scores)
        for dataset_name in model_scores.keys():
            print(f"  Dataset: {dataset_name}")
            dataset_scores = model_scores[dataset_name]
            for method in dataset_scores.keys():
                method_scores = dataset_scores[method]
                print(f"    Method: {method}")
                print(f"      Training samples: {len(method_scores['training'])}")
                print(f"      Non-training samples: {len(method_scores['non-training'])}")

    # Plot distributions
    plot_distributions(scores_data, args.output_dir)

if __name__ == "__main__":
    main()
