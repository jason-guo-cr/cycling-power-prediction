from fitparse import FitFile
from datetime import datetime
import pandas as pd
import os
import glob

def parse_fit_file(fit_file_path, output_csv_path=None):
    """
    解析FIT文件并提取相关骑行数据
    
    参数:
    fit_file_path: FIT文件路径
    output_csv_path: 输出的CSV文件路径，如果为None则不保存文件
    
    返回:
    pandas DataFrame包含解析后的数据
    """
    print(f"解析文件: {fit_file_path}")
    
    # 使用fitparse库读取FIT文件
    fitfile = FitFile(fit_file_path)
    
    # 检查可用字段 - 添加这段代码来检查所有可用字段
    print("\n可用字段检查:")
    available_fields = set()
    sample_record = None
    for i, record in enumerate(fitfile.get_messages('record')):
        if i == 0:
            sample_record = record
        for field in record:
            if field.value is not None:
                available_fields.add(field.name)
        if i > 10:  # 只检查前几条记录即可
            break
    
    print(f"检测到的字段: {sorted(available_fields)}")
    
    # 如果有样本记录，打印详细信息
    if sample_record:
        print("\n样本记录详情:")
        for field in sample_record:
            if field.value is not None:
                print(f"  {field.name}: {field.value} ({type(field.value).__name__})")
    
    # 重新定位到文件开头
    fitfile = FitFile(fit_file_path)
    
    # 提取所有数据记录
    records = []
    for record in fitfile.get_messages('record'):
        data = {}
        # 提取所有可用字段 - 添加enhanced_altitude和enhanced_speed到字段列表
        for field in record:
            if field.name in ['timestamp', 'position_lat', 'position_long', 'distance', 
                             'altitude', 'enhanced_altitude', 'speed', 'enhanced_speed', 'heart_rate', 
                             'cadence', 'temperature', 'power']:
                if field.value is not None:
                    data[field.name] = field.value
        
        if data:  # 只有在有数据的情况下才添加这条记录
            records.append(data)
    
    # 创建DataFrame
    if records:
        df = pd.DataFrame(records)
        
        # 处理enhanced_altitude字段 - 如果没有altitude但有enhanced_altitude，则使用enhanced_altitude
        if 'altitude' not in df.columns and 'enhanced_altitude' in df.columns:
            print("使用enhanced_altitude作为altitude")
            df['altitude'] = df['enhanced_altitude']
            df.drop(['enhanced_altitude'], axis=1, inplace=True)
        elif 'altitude' in df.columns and 'enhanced_altitude' in df.columns:
            # 如果两者都存在，可以检查是否有差异并记录
            if not df['altitude'].equals(df['enhanced_altitude']):
                print("警告: altitude和enhanced_altitude存在差异，使用altitude")
            df.drop(['enhanced_altitude'], axis=1, inplace=True)
        
        # 处理enhanced_speed字段 - 如果没有speed但有enhanced_speed，则使用enhanced_speed
        if 'speed' not in df.columns and 'enhanced_speed' in df.columns:
            print("使用enhanced_speed作为speed")
            df['speed'] = df['enhanced_speed']
            df.drop(['enhanced_speed'], axis=1, inplace=True)
        elif 'speed' in df.columns and 'enhanced_speed' in df.columns:
            # 如果两者都存在，可以检查是否有差异并记录
            if not df['speed'].equals(df['enhanced_speed']):
                print("警告: speed和enhanced_speed存在差异，使用speed")
            df.drop(['enhanced_speed'], axis=1, inplace=True)
        
        # 处理位置数据
        if 'position_lat' in df.columns and 'position_long' in df.columns:
            # 转换GPS坐标从半角度到度
            df['latitude'] = df['position_lat'] / (2**32 / 360)
            df['longitude'] = df['position_long'] / (2**32 / 360)
            df.drop(['position_lat', 'position_long'], axis=1, inplace=True)
        
        # 计算坡度 (如果有海拔和距离数据)
        if 'altitude' in df.columns and 'distance' in df.columns:
            df['altitude_diff'] = df['altitude'].diff()
            df['distance_diff'] = df['distance'].diff()
            # 避免除以零，并计算坡度百分比
            mask = (df['distance_diff'] > 0)
            df['slope'] = 0.0
            df.loc[mask, 'slope'] = (df.loc[mask, 'altitude_diff'] / df.loc[mask, 'distance_diff']) * 100
            
            # 清理临时列
            df.drop(['altitude_diff', 'distance_diff'], axis=1, inplace=True)
        
        # 计算时间差 (秒)
        if 'timestamp' in df.columns:
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
            
            # 添加额外特征 - 如果需要
            # 计算加速度 (如果有速度和时间差)
            if 'speed' in df.columns and 'time_diff' in df.columns:
                mask = (df['time_diff'] > 0)
                df['acceleration'] = 0.0
                df.loc[mask, 'acceleration'] = df.loc[mask, 'speed'].diff() / df.loc[mask, 'time_diff']
            
            # 计算功率变化率
            if 'power' in df.columns and 'time_diff' in df.columns:
                mask = (df['time_diff'] > 0)
                df['power_change_rate'] = 0.0
                df.loc[mask, 'power_change_rate'] = df.loc[mask, 'power'].diff() / df.loc[mask, 'time_diff']
        
        # 保存到CSV (如果指定了输出路径)
        if output_csv_path is not None:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            print(f"保存CSV到: {output_csv_path}")
        
        return df
    else:
        print("未找到记录数据")
        return None

def display_ride_summary(df, filename):
    """
    显示骑行数据的摘要信息
    
    参数:
        df: 包含骑行数据的DataFrame
        filename: 文件名
    """
    if df.empty:
        print(f"{filename}: 没有找到骑行数据")
        return
    
    print(f"\n{filename} 骑行数据摘要:")
    print("-" * 50)
    
    # 计算骑行时间
    if 'timestamp' in df.columns:
        duration = df['timestamp'].max() - df['timestamp'].min()
        print(f"骑行时间: {duration}")
    
    # 显示平均速度
    if 'speed' in df.columns:
        avg_speed = df['speed'].mean() * 3.6  # 转换为km/h
        print(f"平均速度: {avg_speed:.2f} km/h")
    
    # 显示最大速度
    if 'speed' in df.columns:
        max_speed = df['speed'].max() * 3.6  # 转换为km/h
        print(f"最大速度: {max_speed:.2f} km/h")
    
    # 显示总距离
    if 'distance' in df.columns:
        total_distance = df['distance'].max() / 1000  # 转换为公里
        print(f"总距离: {total_distance:.2f} km")
    
    # 显示平均心率
    if 'heart_rate' in df.columns:
        avg_hr = df['heart_rate'].mean()
        print(f"平均心率: {avg_hr:.0f} bpm")
    
    print("-" * 50)

def process_fit_directory(directory_path, output_directory=None):
    """
    处理目录中的所有.fit文件
    
    参数:
        directory_path: 包含.fit文件的目录路径
        output_directory: 输出CSV文件的目录路径（可选）
    """
    if output_directory is None:
        output_directory = directory_path
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 获取所有.fit文件
    fit_files = glob.glob(os.path.join(directory_path, "*.fit"))
    
    if not fit_files:
        print(f"在 {directory_path} 中没有找到.fit文件")
        return
    
    print(f"找到 {len(fit_files)} 个.fit文件")
    
    # 处理每个文件
    for fit_path in fit_files:
        filename = os.path.basename(fit_path)
        print(f"\n正在处理: {filename}")
        
        try:
            # 解析fit文件
            df = parse_fit_file(fit_path)
            
            # 检查解析后的数据框架 - 添加这一部分
            print("\n解析后的CSV数据:")
            print(f"数据行数: {len(df) if df is not None else 0}")
            if df is not None and not df.empty:
                print(f"数据列: {df.columns.tolist()}")
                print("\n数据预览:")
                print(df.head(3))
                print("\n数据统计:")
                print(df.describe().T)
            
            # 生成输出文件名（将.fit替换为.csv）
            output_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_directory, output_filename)
            
            # 保存为CSV文件
            df.to_csv(output_path, index=False)
            print(f"已保存到: {output_path}")
            
            # 显示骑行摘要
            display_ride_summary(df, filename)
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

def main():
    # 设置数据目录
    data_directory = "E:\\open-code\\flash-idea\\sport-data-analysis\\source-data\\test\\fit"  # 替换为你的数据目录路径
    output_directory = "E:\\open-code\\flash-idea\\sport-data-analysis\\source-data\\test"  # 替换为你想要保存CSV文件的目录路径
    
    try:
        # 处理目录中的所有.fit文件
        process_fit_directory(data_directory, output_directory)
        
    except Exception as e:
        print(f"处理目录时出错: {str(e)}")

if __name__ == "__main__":
    main() 