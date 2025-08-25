import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lstm import LSTMModel, load_and_preprocess_data, test_model

def plot_specific_fid_comparison(fid_values, actual_values, predicted_values, step_size, save_path,
                                 figsize=(12, 6), title='Actual vs Predicted AI Values',
                                 xlabel='FID (Time Index)', ylabel='AI Value', show_grid=True):
    """
    绘制预测值和真实值对比图，按指定步长选择FID值

    参数:
    fid_values: FID值数组(空间单元标识)
    actual_values: 实际AI值数组
    predicted_values: 预测AI值数组
    step_size: 步长，用于选择FID值
    save_path: 图片保存路径
    figsize: 图表大小元组 (宽度, 高度)
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    show_grid: 是否显示网格线
    """
    # 创建图表
    plt.figure(figsize=figsize)

    # 计算要显示的FID值
    # 从第一个FID开始，每隔step_size选择一个FID
    selected_indices = np.arange(0, len(fid_values), step_size)
    selected_fids = [fid_values[i] for i in selected_indices if i < len(actual_values)]

    # 绘制每个选中的FID值对应的预测值和真实值
    for i, idx in enumerate(selected_indices):
        if idx >= len(actual_values) or idx >= len(predicted_values):
            continue

        fid = fid_values[idx]
        actual = actual_values[idx]
        predicted = predicted_values[idx]

        # 使用蓝色(o)表示真实值，红色(x)表示预测值
        plt.scatter(fid, actual, color='blue', marker='o', s=100, alpha=0.8, label='Actual' if i == 0 else "")
        plt.scatter(fid, predicted, color='red', marker='x', s=100, alpha=0.8, label='Predicted' if i == 0 else "")

        # 绘制连接线，直观展示预测误差
        plt.plot([fid, fid], [actual, predicted], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 显示网格
    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    # 添加图例
    plt.legend(loc='best', fontsize=12)

    # 自动调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path)
    plt.close()

    print(f"Comparison results saved to: {save_path}")
    print(f"Step size: {step_size}, Total points: {len(selected_fids)}")

# 配置参数
config = {
    'seq_length': 10,                    # 序列长度：用于构建时间序列数据
    'input_size': 7,                     # 输入特征维度：使用7个生态特征
    'hidden_layer_size': 100,            # 隐藏层大小：LSTM隐藏层神经元数量
    'output_size': 1,                    # 输出维度：预测一个值(AI)
    'num_layers': 5,                     # LSTM层数：堆叠LSTM层数
    'batch_size': 64,                    # 批次大小：每次训练样本数
    'model_path': './model/lstm_best_model.pth',  # 预训练模型路径
    'test_path': './datasets/test.csv',  # 测试数据路径
    'comparison_path': './report/fid_comparison.png',  # 对比图保存路径
    'step_size': 100,                    # 步长：控制显示哪些FID值
}

def main():
    # 加载和预处理数据
    # 注意：这里两个参数都使用测试数据路径，因为我们只需要测试数据
    _, _, test_data, scaler, ai_index, test_fid = load_and_preprocess_data(
        config['test_path'],  # 使用测试数据作为训练数据路径
        config['test_path'],  # 使用测试数据作为测试数据路径
        config['seq_length']
    )

    # 创建模型
    model = LSTMModel(
        input_dim=config['input_size'],
        hidden_dim=config['hidden_layer_size'],
        num_layers=config['num_layers'],
        output_dim=config['output_size']
    )

    # 使用预训练模型进行测试
    predictions_real, actual_values_real = test_model(
        model,
        test_data,
        scaler,
        config['seq_length'],
        config['model_path'],
        ai_index
    )

    # 绘制预测值与真实值的对比图
    plot_specific_fid_comparison(
        test_fid,                  # FID值(空间单元标识)
        actual_values_real,        # 实际AI值
        predictions_real,          # 预测AI值
        config['step_size'],       # 步长：控制显示密度
        config['comparison_path']  # 保存路径
    )

    print(f"Predictions completed! Comparison plot saved to {config['comparison_path']}")

if __name__ == "__main__":
    main()