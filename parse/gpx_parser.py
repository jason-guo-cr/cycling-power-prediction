import gpxpy
import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_gpx_file(gpx_file_path):
    """
    解析GPX文件并返回骑行数据
    
    参数:
        gpx_file_path: GPX文件的路径
    返回:
        包含骑行数据的DataFrame
    """
    # 读取GPX文件
    with open(gpx_file_path, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    # 用于存储所有点的数据
    data_points = []
    
    # 遍历所有track点
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data_point = {
                    'timestamp': point.time,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'altitude': point.elevation,
                    'distance': 0,  # 将在后面计算
                    'speed': 0,     # 将在后面计算
                }
                data_points.append(data_point)
    
    # 转换为DataFrame
    df = pd.DataFrame(data_points)
    
    # 计算累计距离和速度
    if len(df) > 0:
        # 确保时间戳存在
        if df['timestamp'].isna().any():
            df['timestamp'] = pd.date_range(
                start=datetime.now(),
                periods=len(df),
                freq='1S'
            )
        
        # 计算两点之间的距离（使用Haversine公式）
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # 地球半径（米）
            
            # 转换为弧度
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine公式
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            return distance
        
        # 计算累计距离
        df['distance'] = 0.0
        for i in range(1, len(df)):
            distance = haversine_distance(
                df['latitude'].iloc[i-1],
                df['longitude'].iloc[i-1],
                df['latitude'].iloc[i],
                df['longitude'].iloc[i]
            )
            df.loc[i, 'distance'] = df['distance'].iloc[i-1] + distance
        
        # 计算速度 (m/s)
        df['speed'] = 0.0
        time_diff = df['timestamp'].diff().dt.total_seconds()
        distance_diff = df['distance'].diff()
        mask = time_diff > 0
        df.loc[mask, 'speed'] = distance_diff[mask] / time_diff[mask]
        
        # 添加其他可能的字段，初始值设为NaN
        df['heart_rate'] = np.nan
        df['cadence'] = np.nan
        df['temperature'] = np.nan
        
        # 重命名列以匹配FIT文件格式
        df = df.rename(columns={
            'latitude': 'position_lat',
            'longitude': 'position_long'
        })
    
    return df

def process_gpx_directory(directory_path, output_directory=None):
    """
    处理目录中的所有GPX文件
    
    参数:
        directory_path: 包含GPX文件的目录路径
        output_directory: 输出CSV文件的目录路径（可选）
    """
    if output_directory is None:
        output_directory = directory_path
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 处理所有GPX文件
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.gpx'):
            gpx_path = os.path.join(directory_path, filename)
            print(f"正在处理: {filename}")
            
            try:
                # 解析GPX文件
                df = parse_gpx_file(gpx_path)
                
                # 生成输出文件名
                output_filename = os.path.splitext(filename)[0] + '.csv'
                output_path = os.path.join(output_directory, output_filename)
                
                # 保存为CSV
                df.to_csv(output_path, index=False)
                print(f"已保存到: {output_path}")
                
                # 显示基本统计信息
                if len(df) > 0:
                    duration = df['timestamp'].max() - df['timestamp'].min()
                    total_distance = df['distance'].max() / 1000  # 转换为公里
                    avg_speed = df['speed'].mean() * 3.6  # 转换为km/h
                    
                    print("\n骑行数据摘要:")
                    print("-" * 30)
                    print(f"总时长: {duration}")
                    print(f"总距离: {total_distance:.2f} km")
                    print(f"平均速度: {avg_speed:.2f} km/h")
                    print(f"数据点数: {len(df)}")
                    print("-" * 30)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

def main():
    # 示例用法
    gpx_file_path = "E:\\open-code\\flash-idea\\sport-data-analysis\\source-data\\2025-03-05-morning.gpx"
    
    try:
        # 解析单个GPX文件
        df = parse_gpx_file(gpx_file_path)
        
        # 保存为CSV文件
        output_path = "ride_data_from_gpx.csv"
        df.to_csv(output_path, index=False)
        print(f"\n数据已保存到 {output_path}")
        
        # 显示基本统计信息
        if len(df) > 0:
            duration = df['timestamp'].max() - df['timestamp'].min()
            total_distance = df['distance'].max() / 1000  # 转换为公里
            avg_speed = df['speed'].mean() * 3.6  # 转换为km/h
            
            print("\n骑行数据摘要:")
            print("-" * 30)
            print(f"总时长: {duration}")
            print(f"总距离: {total_distance:.2f} km")
            print(f"平均速度: {avg_speed:.2f} km/h")
            print(f"数据点数: {len(df)}")
        
    except Exception as e:
        print(f"解析文件时出错: {str(e)}")

if __name__ == "__main__":
    main() 