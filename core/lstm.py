import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mplfinance as mpf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
import json


# 设置 pandas 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 设置随机种子，保证结果可复现
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# ================================
# 配置超参数
# ================================
config = {
    'seq_length': 10,                 # 序列长度：用于构建时间序列数据
    'input_size': 7,                  # 输入特征维度：使用7个生态特征
    'hidden_layer_size': 100,         # 隐藏层大小：LSTM隐藏层神经元数量
    'output_size': 1,                 # 输出维度：预测一个值(AI)
    'batch_size': 64,                 # 批次大小：每次训练样本数
    'learning_rate': 0.001,           # 学习率：控制参数更新步长
    'num_layers': 5,                  # LSTM层数：堆叠LSTM层数
    'epochs': 150,                    # 训练轮数：完整训练数据集次数
    'val_split': 0.2,                 # 验证集比例：训练数据中用于验证的比例
    'accuracy_threshold': 0.01,       # 准确率阈值：判断预测是否准确的阈值
    'train_path': './datasets/train.csv',  # 训练数据路径
    'test_path': './datasets/test.csv',    # 测试数据路径
    'best_model_path': './model/lstm_best_model.pth',  # 最佳模型保存路径
    'json_path': './report/lstm_train_report.json',     # 训练报告保存路径
    'save_png': './report/lstm_report_table.png',       # 训练指标图表保存路径
    'save_pred_png': './report/lstm_prediction.png'     # 预测结果图表保存路径
}


# ================================
# 计算 MSE 和 RMSE
# ================================
def calculate_mse_rmse(predictions, targets, scaler, ai_index):
    """
    计算均方误差(MSE)和均方根误差(RMSE)

    参数:
    predictions: 预测值张量
    targets: 实际值张量
    scaler: 归一化器，用于反归一化
    ai_index: AI值在特征中的索引位置

    返回:
    mse: 均方误差
    rmse: 均方根误差
    """
    # 反归一化：将归一化后的值恢复到原始范围
    predictions_real = predictions * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]
    targets_real = targets * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]

    # 计算均方误差（MSE）
    mse = torch.mean((predictions_real - targets_real) ** 2)

    # 计算均方根误差（RMSE）
    rmse = torch.sqrt(mse)

    return mse.item(), rmse.item()


# ================================
# 计算准确率
# ================================
def calculate_accuracy(predictions, targets, threshold, scaler, ai_index):
    """
    计算预测准确率：预测值与实际值的误差在阈值范围内的比例

    参数:
    predictions: 预测值张量
    targets: 实际值张量
    threshold: 准确率阈值
    scaler: 归一化器，用于反归一化
    ai_index: AI值在特征中的索引位置

    返回:
    accuracy: 准确率
    """
    # 计算归一化前的实际阈值
    threshold_real = threshold * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index])

    # 反归一化
    predictions_real = predictions * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]
    targets_real = targets * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]

    # 计算绝对误差
    abs_error = torch.abs(predictions_real - targets_real)

    # 计算误差在阈值范围内的比例作为准确率
    accuracy = (abs_error <= threshold_real).float().mean()
    return accuracy.item()


# ================================
# 数据预处理函数
# ================================
def load_and_preprocess_data(train_path, test_path, seq_length):
    """
    加载并预处理数据，构建时间序列数据集

    参数:
    train_path: 训练数据路径
    test_path: 测试数据路径
    seq_length: 序列长度

    返回:
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    test_data: 测试数据
    scaler: 归一化器
    ai_index: AI值在特征中的索引
    test_fid: 测试数据的FID（时间标识）
    """
    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 删除含NA值的行
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # 特征列选择，不包含FID（空间单元标识）
    feature_cols = [
        'PD', 'LPI', 'ED', 'LSI', 'CONTAG', 'SHDI', 'SHEI'  # 生态景观特征
    ]
    ai_index = 7  # AI在完整特征列表中的索引位置（7个特征 + AI）

    # 预测目标列：景观聚集度指数(AI)
    target_col = 'AI'

    # 保存FID用于后续绘图（仅作为时间标识）
    train_fid = train_df['FID'].values
    test_fid = test_df['FID'].values

    # 确保数据按FID递增排序，保证时间序列的连续性
    train_df = train_df.sort_values('FID')
    test_df = test_df.sort_values('FID')

    # 提取特征数据
    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values

    # 提取目标数据
    train_target = train_df[target_col].values
    test_target = test_df[target_col].values

    # 归一化 - 对特征和目标一起进行归一化，保证数据尺度一致
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 创建完整的数据集进行归一化
    all_train_data = np.hstack([train_features, train_target.reshape(-1, 1)])
    all_test_data = np.hstack([test_features, test_target.reshape(-1, 1)])

    # 拟合和转换训练数据
    all_train_data_scaled = scaler.fit_transform(all_train_data)

    # 仅使用训练数据的统计信息转换测试数据，避免数据泄露
    all_test_data_scaled = scaler.transform(all_test_data)

    def create_sequences(data):
        """
        将数据转换为序列格式，用于LSTM输入

        参数:
        data: 原始数据

        返回:
        xs: 序列特征
        ys: 目标值
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length, :len(feature_cols)]  # 特征数据（不包含FID）
            y = data[i + seq_length, len(feature_cols)]  # 目标数据(AI)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # 创建序列
    X, y = create_sequences(all_train_data_scaled)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['val_split'], random_state=42, shuffle=True)

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 打印训练集和测试集的前10个样本，用于数据检查
    print("训练集前10个样本：")
    print(train_df.head(10))
    print("\n测试集前10个样本：")
    print(test_df.head(10))

    return train_loader, val_loader, all_test_data_scaled, scaler, ai_index, test_fid


# ================================
# LSTM模型定义
# ================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层：处理序列数据
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接层：将LSTM输出映射到预测值
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # 只取序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_loader, val_loader, epochs, lr, threshold, best_model_path, scaler, ai_index,
                json_path=config['json_path']):
    """
    训练LSTM模型并保存最佳模型

    参数:
    model: LSTM模型
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    epochs: 训练轮数
    lr: 学习率
    threshold: 准确率阈值
    best_model_path: 最佳模型保存路径
    scaler: 归一化器
    ai_index: AI值在特征中的索引
    json_path: 训练指标保存路径
    """
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -float('inf')  # 初始化最好的验证集准确率

    # 创建字典来存储每个epoch的指标，用于后续分析
    metrics = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "train_mse": [],
        "train_rmse": [],
        "val_loss": [],
        "val_acc": [],
        "val_mse": [],
        "val_rmse": []
    }

    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss, train_acc, train_mse, train_rmse = 0.0, 0.0, 0.0, 0.0

        # 遍历训练数据批次
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output.squeeze(), y_batch)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            # 累积训练指标
            train_loss += loss.item()
            train_acc += calculate_accuracy(output.squeeze(), y_batch, threshold, scaler, ai_index)
            mse, rmse = calculate_mse_rmse(output.squeeze(), y_batch, scaler, ai_index)
            train_mse += mse
            train_rmse += rmse

        # 验证模式
        model.eval()
        val_loss, val_acc, val_mse, val_rmse = 0.0, 0.0, 0.0, 0.0

        # 遍历验证数据批次
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = loss_fn(output.squeeze(), y_batch)

                # 累积验证指标
                val_loss += loss.item()
                val_acc += calculate_accuracy(output.squeeze(), y_batch, threshold, scaler, ai_index)
                mse, rmse = calculate_mse_rmse(output.squeeze(), y_batch, scaler, ai_index)
                val_mse += mse
                val_rmse += rmse

        # 如果当前验证集准确率更好，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())  # 保存当前模型
            torch.save(best_model, best_model_path)  # 保存最佳模型

        # 保存指标到字典
        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_acc"].append(train_acc / len(train_loader))
        metrics["train_mse"].append(train_mse / len(train_loader))
        metrics["train_rmse"].append(train_rmse / len(train_loader))
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["val_acc"].append(val_acc / len(val_loader))
        metrics["val_mse"].append(val_mse / len(val_loader))
        metrics["val_rmse"].append(val_rmse / len(val_loader))

        # 打印当前轮次的训练和验证指标
        print(f'Epoch [{epoch + 1}/{epochs}] - '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc / len(train_loader):.4f}, '
              f'Train MSE: {train_mse / len(train_loader):.4f}, '
              f'Train RMSE: {train_rmse / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc / len(val_loader):.4f}, '
              f'Val MSE: {val_mse / len(val_loader):.4f}, '
              f'Val RMSE: {val_rmse / len(val_loader):.4f}')

    # 将训练指标保存到JSON文件
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def test_model(model, test_data, scaler, seq_length, best_model_path, ai_index):
    """
    使用最佳模型进行测试并评估性能

    参数:
    model: LSTM模型
    test_data: 测试数据
    scaler: 归一化器
    seq_length: 序列长度
    best_model_path: 最佳模型路径
    ai_index: AI值在特征中的索引

    返回:
    predictions_real: 反归一化后的预测值
    actual_values_real: 反归一化后的实际值
    """
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 准备测试序列
    test_sequences = []
    for i in range(len(test_data) - seq_length):
        test_sequences.append(test_data[i:i + seq_length, :config['input_size']])  # 只取特征部分
    test_sequences = np.array(test_sequences)

    # 转换为张量并进行预测
    test_tensor = torch.tensor(test_sequences, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).squeeze().numpy()

    # 反归一化预测值
    predictions_real = predictions * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]

    # 获取实际值
    actual_values = test_data[seq_length:, ai_index]  # 真实的AI值
    actual_values_real = actual_values * (scaler.data_max_[ai_index] - scaler.data_min_[ai_index]) + scaler.data_min_[ai_index]

    # 计算测试集MSE和RMSE
    mse, rmse = calculate_mse_rmse(torch.tensor(predictions), torch.tensor(actual_values), scaler, ai_index)

    # 打印前10个预测结果和实际值
    print("Test Data (First 10 Samples) and Predictions:")
    for i in range(10):
        print(f"Test Data {i + 1}: {actual_values_real[i]:.4f} -> Prediction: {predictions_real[i]:.4f}")

    # 计算准确率
    accuracy = calculate_accuracy(torch.tensor(predictions), torch.tensor(actual_values),
                                  config['accuracy_threshold'], scaler, ai_index) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    return predictions_real, actual_values_real


# Function to plot metrics
def plot_metrics(json_path):
    """
    绘制训练指标图表

    参数:
    json_path: 训练指标JSON文件路径
    """
    # 加载训练指标
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 绘制损失曲线
    axes[0, 0].plot(metrics["epochs"], metrics["train_loss"], label="Train Loss", color='b')
    axes[0, 0].plot(metrics["epochs"], metrics["val_loss"], label="Val Loss", color='r')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    # 绘制准确率曲线
    axes[0, 1].plot(metrics["epochs"], metrics["train_acc"], label="Train Accuracy", color='b')
    axes[0, 1].plot(metrics["epochs"], metrics["val_acc"], label="Val Accuracy", color='r')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    # 绘制MSE曲线
    axes[1, 0].plot(metrics["epochs"], metrics["train_mse"], label="Train MSE", color='b')
    axes[1, 0].plot(metrics["epochs"], metrics["val_mse"], label="Val MSE", color='r')
    axes[1, 0].set_title('MSE')
    axes[1, 0].legend()

    # 绘制RMSE曲线
    axes[1, 1].plot(metrics["epochs"], metrics["train_rmse"], label="Train RMSE", color='b')
    axes[1, 1].plot(metrics["epochs"], metrics["val_rmse"], label="Val RMSE", color='r')
    axes[1, 1].set_title('RMSE')
    axes[1, 1].legend()

    # 保存图表
    plt.savefig(config['save_png'])  # Save the figure as PNG file


def plot_prediction_results(fid_values, actual_values, predicted_values, save_path):
    """
    绘制预测结果与实际值的对比图

    参数:
    fid_values: FID值数组（时间标识）
    actual_values: 实际AI值数组
    predicted_values: 预测AI值数组
    save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(fid_values[config['seq_length']:], actual_values, label='Actual AI', color='blue')
    plt.plot(fid_values[config['seq_length']:], predicted_values, label='Predicted AI', color='red', linestyle='--')
    plt.title('Actual vs Predicted AI Values')
    plt.xlabel('FID (Time Index)')
    plt.ylabel('AI Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # 更新输入大小配置
    config['input_size'] = 7  # 7个特征（不包括FID）

    # 加载和预处理数据
    train_loader, val_loader, test_data, scaler, ai_index, test_fid = load_and_preprocess_data(
        config['train_path'], config['test_path'], config['seq_length'])

    # 创建模型 - 更新参数以匹配LSTMModel的定义
    model = LSTMModel(
        input_dim=config['input_size'],
        hidden_dim=config['hidden_layer_size'],
        num_layers=config['num_layers'],  # 添加缺失的num_layers参数
        output_dim=config['output_size']
    )

    # 训练模型
    train_model(model, train_loader, val_loader,
                epochs=config['epochs'],
                lr=config['learning_rate'],
                threshold=config['accuracy_threshold'],
                best_model_path=config['best_model_path'],
                scaler=scaler, ai_index=ai_index)

    # 绘制训练指标
    plot_metrics(config['json_path'])

    # 测试模型
    predictions_real, actual_values_real = test_model(model, test_data, scaler, config['seq_length'],
                                                      config['best_model_path'], ai_index)

    # 绘制预测结果
    plot_prediction_results(test_fid, actual_values_real, predictions_real, config['save_pred_png'])

    # 打印完成信息
    print(f"训练和预测完成！指标图表已保存到 {config['save_png']}")
    print(f"预测结果图表已保存到 {config['save_pred_png']}")
    print(f"最佳模型已保存到 {config['best_model_path']}")