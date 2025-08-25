import pandas as pd
import sys
import os

def process_csv_files(file_paths, output_path):
    # 存储处理后的数据框列表
    dfs = []

    # 处理每个CSV文件
    for file_path in file_paths:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            continue

        try:
            # 读取CSV文件
            print(f"正在读取文件: {file_path}")
            df = pd.read_csv(file_path)

            # 去除列名中的空格
            df.columns = df.columns.str.strip()

            # 打印实际列名用于调试
            print(f"文件 '{file_path}' 的列名: {list(df.columns)}")

            # 检查必要的列是否存在
            required_columns = ['FID', 'PD', 'LPI', 'ED', 'LSI', 'CONTAG', 'SHDI', 'SHEI']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"错误: 文件 '{file_path}' 缺少必要的列: {', '.join(missing_columns)}")
                continue

            # 去除字符串列中的空格（可选步骤，根据数据情况决定是否需要）
            # 注意：这会影响所有字符串列，包括可能包含合法空格的文本
            # 如果你有包含合法空格的文本列，请有选择地应用此操作
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

            # 删除指定列中任意一个值为0的行
            filter_columns = ['PD', 'LPI', 'ED', 'LSI', 'CONTAG', 'SHDI', 'SHEI']
            df = df[(df[filter_columns] != 0).all(axis=1)]

            # 将处理后的数据框添加到列表
            dfs.append(df)

            print(f"已处理文件: {file_path}，剩余行数: {len(df)}")

        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错: {str(e)}")

    # 检查是否有数据可合并
    if not dfs:
        print("没有有效数据可合并")
        return False

    # 合并所有数据框
    combined_df = pd.concat(dfs, ignore_index=True)

    # 重置FID列，从0开始递增
    combined_df['FID'] = range(len(combined_df))

    # 保存结果到新的CSV文件
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"合并后的文件已保存到: {output_path}")
        print(f"总行数: {len(combined_df)}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

if __name__ == "__main__":
    input_files = ['./datasets/2015_part2.csv']
    output_file = './datasets/test.csv'
    # 处理CSV文件
    success = process_csv_files(input_files, output_file)

    # 根据处理结果设置退出码
    sys.exit(0 if success else 1)