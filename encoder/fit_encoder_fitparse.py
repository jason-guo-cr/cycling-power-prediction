import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# 安装并导入fit4python
# pip install fit4python
import fit4python

# 添加FIT_UTC_REFERENCE定义
FIT_UTC_REFERENCE = datetime(1989, 12, 31, 0, 0, 0)

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def csv_to_fit(csv_path, output_fit_path=None):
    """使用fit4python库将CSV文件转换为FIT格式"""
    logging.info(f"正在将CSV文件转换为FIT: {csv_path}")
    
    if output_fit_path is None:
        output_fit_path = os.path.splitext(csv_path)[0] + '.fit'
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    logging.info(f"CSV文件读取完成，行数: {len(df)}, 列: {df.columns.tolist()}")
    
    # 检查必要的列
    required_columns = ['timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 转换timestamp为datetime格式（如果不是）
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 创建FIT文件（二进制方式）
    # 这需要理解FIT文件格式并手动构建二进制数据
    
    # 这里是一个简化的示例，实际实现需要更复杂的代码
    with open(output_fit_path, 'wb') as fit_file:
        # 写入文件头
        fit_file.write(b'\x0E\x10\x00\x00\x00\x00\x00\x00\x2E\x46\x49\x54\x00\x00')
        
        # 在这里添加消息定义和消息
        # 这部分需要更详细的FIT文件格式知识
        
    logging.info(f"FIT文件已保存到: {output_fit_path}")
    return output_fit_path

# 以下是使用fitdecode库的新函数实现
def add_file_id_message_new(encoder, df):
    """添加文件 ID 消息"""
    first_timestamp = df['timestamp'].min()
    fit_timestamp = int((first_timestamp - pd.Timestamp(FIT_UTC_REFERENCE)).total_seconds())
    
    file_id = encoder.add_message('file_id')
    file_id.add_field('type', 4)  # 4 = activity
    file_id.add_field('manufacturer', 1)  # 1 = Garmin
    file_id.add_field('product', 20)
    file_id.add_field('serial_number', 0x12345678)
    file_id.add_field('time_created', fit_timestamp)

def add_record_messages_new(encoder, df):
    """添加记录消息"""
    for idx, row in df.iterrows():
        timestamp = int((row['timestamp'] - pd.Timestamp(FIT_UTC_REFERENCE)).total_seconds())
        
        record = encoder.add_message('record')
        record.add_field('timestamp', timestamp)
        
        # 添加位置数据
        if 'latitude' in df.columns and 'longitude' in df.columns:
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                position_lat = int(row['latitude'] * (2**32 / 360))
                position_long = int(row['longitude'] * (2**32 / 360))
                record.add_field('position_lat', position_lat)
                record.add_field('position_long', position_long)
        
        # 添加高度数据
        if 'altitude' in df.columns and pd.notna(row['altitude']):
            altitude = int(row['altitude'] * 5)  # 缩放因子 5
            record.add_field('altitude', altitude)
        
        # 添加心率数据
        if 'heart_rate' in df.columns and pd.notna(row['heart_rate']):
            heart_rate = int(row['heart_rate'])
            record.add_field('heart_rate', heart_rate)
        
        # 添加踏频数据
        if 'cadence' in df.columns and pd.notna(row['cadence']):
            cadence = int(row['cadence'])
            record.add_field('cadence', cadence)
        
        # 添加距离数据
        if 'distance' in df.columns and pd.notna(row['distance']):
            distance = int(row['distance'] * 100)  # 缩放因子 100
            record.add_field('distance', distance)
        
        # 添加速度数据
        if 'speed' in df.columns and pd.notna(row['speed']):
            speed = int(row['speed'] * 1000)  # 缩放因子 1000
            record.add_field('speed', speed)
        
        # 添加功率数据
        if 'power' in df.columns and pd.notna(row['power']):
            power = int(row['power'])
            record.add_field('power', power)
        
        # 添加温度数据
        if 'temperature' in df.columns and pd.notna(row['temperature']):
            temperature = int(row['temperature'])
            record.add_field('temperature', temperature)

def add_session_message_new(encoder, df):
    """添加会话消息"""
    # 计算会话相关值
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration = (end_time - start_time).total_seconds()
    
    # 时间戳转换
    start_timestamp = int((start_time - pd.Timestamp(FIT_UTC_REFERENCE)).total_seconds())
    end_timestamp = int((end_time - pd.Timestamp(FIT_UTC_REFERENCE)).total_seconds())
    
    # 计算统计值
    total_distance = 0
    avg_speed = 0
    max_speed = 0
    avg_hr = 0
    max_hr = 0
    avg_power = 0
    max_power = 0
    
    if 'distance' in df.columns and not df['distance'].isna().all():
        total_distance = int(df['distance'].dropna().max() * 100)  # 缩放因子 100
    
    if 'speed' in df.columns and not df['speed'].isna().all():
        avg_speed = int(df['speed'].dropna().mean() * 1000)  # 缩放因子 1000
        max_speed = int(df['speed'].dropna().max() * 1000)
    
    if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
        avg_hr = int(df['heart_rate'].dropna().mean())
        max_hr = int(df['heart_rate'].dropna().max())
    
    if 'power' in df.columns and not df['power'].isna().all():
        avg_power = int(df['power'].dropna().mean())
        max_power = int(df['power'].dropna().max())
    
    # 创建会话消息
    session = encoder.add_message('session')
    session.add_field('timestamp', end_timestamp)
    session.add_field('start_time', start_timestamp)
    session.add_field('total_elapsed_time', int(duration * 1000))  # 转换为毫秒
    session.add_field('total_timer_time', int(duration * 1000))  # 转换为毫秒
    session.add_field('total_distance', total_distance)
    session.add_field('sport', 2)  # 2 = cycling
    
    if avg_speed > 0:
        session.add_field('avg_speed', avg_speed)
    
    if max_speed > 0:
        session.add_field('max_speed', max_speed)
    
    if avg_hr > 0:
        session.add_field('avg_heart_rate', avg_hr)
    
    if max_hr > 0:
        session.add_field('max_heart_rate', max_hr)
    
    if avg_power > 0:
        session.add_field('avg_power', avg_power)
    
    if max_power > 0:
        session.add_field('max_power', max_power)

def add_activity_message_new(encoder, df):
    """添加活动消息"""
    # 计算活动相关值
    end_time = df['timestamp'].max()
    start_time = df['timestamp'].min()
    duration = (end_time - start_time).total_seconds()
    
    # 时间戳转换
    end_timestamp = int((end_time - pd.Timestamp(FIT_UTC_REFERENCE)).total_seconds())
    
    # 创建活动消息
    activity = encoder.add_message('activity')
    activity.add_field('timestamp', end_timestamp)
    activity.add_field('total_timer_time', int(duration * 1000))  # 转换为毫秒
    activity.add_field('num_sessions', 1)
    activity.add_field('type', 0)  # 0 = manual
    activity.add_field('event', 26)  # 26 = activity
    activity.add_field('event_type', 1)  # 1 = stop
    activity.add_field('local_timestamp', end_timestamp)

def process_csv_directory(directory_path, output_directory=None):
    """
    处理目录中的所有 CSV 文件并转换为 FIT 格式
    
    参数:
        directory_path: 包含 CSV 文件的目录路径
        output_directory: 输出 FIT 文件的目录路径（可选）
    """
    if output_directory is None:
        output_directory = os.path.join(directory_path, 'fit_encoded')
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        logging.warning(f"在 {directory_path} 中没有找到 CSV 文件")
        return
    
    logging.info(f"找到 {len(csv_files)} 个 CSV 文件")
    processed_count = 0
    error_count = 0
    
    # 处理每个文件
    for csv_filename in csv_files:
        csv_path = os.path.join(directory_path, csv_filename)
        logging.info(f"\n正在处理: {csv_filename}")
        
        # 生成输出文件名（将 .csv 替换为 .fit）
        fit_filename = os.path.splitext(csv_filename)[0] + '.fit'
        fit_path = os.path.join(output_directory, fit_filename)
        
        try:
            # 转换 CSV 到 FIT
            fit_path = csv_to_fit(csv_path, fit_path)
            
            logging.info(f"文件创建成功: {fit_filename}")
            processed_count += 1
            
        except Exception as e:
            logging.error(f"处理文件 {csv_filename} 时出错: {str(e)}")
            error_count += 1
    
    logging.info(f"\n处理完成: 成功处理 {processed_count} 个文件，错误 {error_count} 个文件")

def main():
    # 设置数据目录
    data_directory = "E:\\open-code\\flash-idea\\cycling-power-prediction\\encoder\\data\\csv"  # 替换为你的CSV数据目录路径
    output_directory = "E:\\open-code\\flash-idea\\cycling-power-prediction\\encoder\\data\\fit"  # 替换为你想要保存FIT文件的目录路径
    
    try:
        # 处理目录中的所有CSV文件
        process_csv_directory(data_directory, output_directory)
        
    except Exception as e:
        logging.error(f"处理目录时出错: {str(e)}")

if __name__ == "__main__":
    main() 