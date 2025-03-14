import os
import pandas as pd
import numpy as np
import joblib
import argparse
import json
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_model(model_path='model/power_prediction_model.pkl'):
    """
    加载保存的模型和特征列表
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model_info = joblib.load(model_path)
    model = model_info['model']
    features = model_info['features']
    transform_power = model_info.get('transform_power', False)  # 获取是否进行了功率变换
    
    return model, features, transform_power

def extract_rider_id(filename):
    """
    从文件名中提取骑行者ID
    格式：rider前缀到第一个下划线之间的部分
    例如：rider1_xxx.csv -> rider1
    """
    basename = os.path.basename(filename)
    if basename.startswith('rider'):
        parts = basename.split('_', 1)
        if len(parts) > 0:
            return parts[0]
    return None

def load_rider_info(rider_id, rider_info_path='source-data/rider/rider_info.json'):
    """
    从rider_info.json文件中获取骑行者信息，支持多种可能的路径
    """
    if not rider_id:
        return None
        
    try:
        # 尝试多种可能的路径
        possible_paths = [
            rider_info_path,
            os.path.join(os.path.dirname(__file__), '..', rider_info_path),
            os.path.abspath(rider_info_path),
            'source-data/rider/rider_info.json',  # 相对于当前工作目录
            os.path.join(os.path.dirname(__file__), '..', 'source-data/rider/rider_info.json'),  # 相对于脚本所在目录
        ]
        
        # 尝试所有可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到骑行者信息文件: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    rider_data = json.load(f)
                    if rider_id in rider_data:
                        print(f"获取到骑行者 {rider_id} 的信息: {rider_data[rider_id]}")
                        return rider_data[rider_id]
                    else:
                        print(f"警告: 在JSON文件中未找到骑行者ID: {rider_id}")
                        print(f"可用的骑行者ID: {list(rider_data.keys())}")
                break
        else:
            print(f"警告: 无法找到骑行者信息文件，尝试了以下路径:")
            for path in possible_paths:
                print(f"  - {path}")
            print("请确保rider_info.json文件存在于正确位置")
    except Exception as e:
        print(f"加载骑行者信息时出错: {e}")
    
    return None

def predict_from_csv(model, features, csv_path, rider_info=None, transform_power=False):
    """
    对单个CSV文件进行功率预测
    
    参数:
    model: 训练好的模型
    features: 模型使用的特征列表
    csv_path: CSV文件路径
    rider_info: 包含骑行者信息的字典，如 {'height': 175, 'weight': 70}
    transform_power: 是否需要对预测结果进行逆对数变换
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 如果没有提供骑行者信息，尝试从文件名提取
    if rider_info is None:
        rider_id = extract_rider_id(csv_path)
        if rider_id:
            print(f"从文件名 {os.path.basename(csv_path)} 中提取到骑行者ID: {rider_id}")
            rider_info = load_rider_info(rider_id)
    
    # 加载CSV数据
    df = pd.read_csv(csv_path)
    
    # 添加骑行者信息（如果提供）
    if rider_info is not None:
        print(f"使用骑行者信息: {rider_info}")
        if 'height' in rider_info and 'height' in features:
            df['height'] = rider_info['height']
            print(f"添加身高信息: {rider_info['height']}cm")
        
        if 'weight' in rider_info and 'weight' in features:
            df['weight'] = rider_info['weight']
            print(f"添加体重信息: {rider_info['weight']}kg")
    
    # 准备特征数据前检查关键特征是否缺失
    has_critical_missing_features = False
    critical_features = ['speed', 'cadence']  # 定义关键特征
    
    # 检查数据中是否存在这些关键特征
    critical_missing = [feature for feature in critical_features 
                      if feature in features and feature not in df.columns]
    
    # 标记整个数据集是否完全缺少关键特征
    if critical_missing:
        print(f"警告: 数据集完全缺少关键特征: {critical_missing}")
        has_critical_missing_features = True
    
    # 检查每行数据是否存在缺失的关键特征
    missing_mask = pd.Series(False, index=df.index)
    for feature in critical_features:
        if feature in df.columns:
            missing_mask = missing_mask | df[feature].isna()
    
    if missing_mask.any():
        print(f"警告: 发现 {missing_mask.sum()} 行数据存在缺失的关键特征")
    
    # 添加相互影响特征
    # 如果有speed和cadence，添加speed_cadence交互特征
    if 'speed' in df.columns and 'cadence' in df.columns and 'speed_cadence' in features:
        df['speed_cadence'] = df['speed'].fillna(0) * df['cadence'].fillna(0)
        print("添加速度-踏频交互特征")
    
    # 添加心率的平方作为特征
    if 'heart_rate' in df.columns and 'heart_rate_squared' in features:
        df['heart_rate_squared'] = df['heart_rate'].fillna(0) ** 2
        print("添加心率平方特征")
    
    # 添加速度的立方项
    if 'speed' in df.columns and 'speed_cubed' in features:
        df['speed_cubed'] = df['speed'].fillna(0) ** 3
        print("添加速度立方特征")
    
    # 确保所有特征都存在
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"警告: CSV文件缺少特征: {missing_features}")
        for feature in missing_features:
            df[feature] = 0  # 用0填充缺失特征
    
    # 准备特征数据
    X = df[features]
    
    # 进行预测
    df['predicted_power'] = model.predict(X)
    
    # 如果进行了对数变换，需要进行逆变换
    if transform_power:
        df['predicted_power'] = np.expm1(df['predicted_power'])
        print("已对预测功率进行指数逆变换")
    
    # 应用物理模型约束
    df = apply_physics_constraints(df, rider_info)
    
    # 确保所有功率非负（放在物理模型约束之后）
    neg_power_count = (df['predicted_power'] < 0).sum()
    if neg_power_count > 0:
        print(f"警告: 物理约束后仍检测到 {neg_power_count} 个负值功率预测，已修正为0")
        df['predicted_power'] = df['predicted_power'].clip(lower=0)
    
    # 如果存在真实功率值，计算评估指标
    if 'power' in df.columns:
        rmse = np.sqrt(mean_squared_error(df['power'], df['predicted_power']))
        r2 = r2_score(df['power'], df['predicted_power'])
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")
        
        # 创建对比图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        time_col = 'timestamp' if 'timestamp' in df.columns else df.index
        plt.plot(time_col, df['power'], 'b-', label='实际功率')
        plt.plot(time_col, df['predicted_power'], 'r-', label='预测功率')
        plt.title('功率对比 (时间序列)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['power'], df['predicted_power'], alpha=0.5)
        plt.plot([df['power'].min(), df['power'].max()], 
                [df['power'].min(), df['power'].max()], 'r--')
        plt.xlabel('实际功率')
        plt.ylabel('预测功率')
        plt.title('功率预测对比')
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('view', exist_ok=True)
        output_filename = os.path.basename(csv_path).replace('.csv', '_prediction.png')
        plt.savefig(f'view/{output_filename}')
    
    # 保存预测结果
    output_dir = os.path.join('prediction', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(csv_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_filename = f"{filename_without_ext}_predicted.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"预测结果保存到: {output_path}")
    
    return df

def batch_predict(model, features, input_dir, rider_info_path='source-data/rider/rider_info.json', transform_power=False):
    """
    批量处理目录中的所有CSV文件
    
    参数:
    model: 训练好的模型
    features: 模型使用的特征列表
    input_dir: 包含CSV文件的目录
    rider_info_path: 骑行者信息文件路径
    transform_power: 是否需要对预测结果进行逆对数变换
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    results = []
    count = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(root, file)
                    print(f"\n处理文件: {file_path}")
                    
                    # 直接在predict_from_csv函数中处理rider_info提取
                    result = predict_from_csv(model, features, file_path, transform_power=transform_power)
                    results.append((file_path, result))
                    count += 1
                except Exception as e:
                    print(f"处理 {file} 时出错: {e}")
    
    print(f"已处理 {count} 个CSV文件")
    return results

def apply_physics_constraints(df, rider_info=None):
    """应用物理规则约束功率预测"""
    # 获取骑行者体重
    weight_kg = rider_info.get('weight', 70) if rider_info else 70
    
    # 1. 速度为0或缺失时，功率应很低
    if 'speed' in df.columns:
        zero_speed = (df['speed'].isna()) | (df['speed'] <= 0.1)
        if zero_speed.any():
            # 静止状态下的基础功率 (约0.5W/kg)
            baseline_power = weight_kg * 0.5
            df.loc[zero_speed, 'predicted_power'] = baseline_power
    
    # 2. 对于有速度数据的行，应用简化的物理模型进行合理性检查
    if 'speed' in df.columns and 'predicted_power' in df.columns:
        # 标识有效速度的行
        has_speed = (~df['speed'].isna()) & (df['speed'] > 0.1)
        
        if has_speed.any():
            # 常量
            CdA = 0.4  # 空气阻力系数
            Crr = 0.005  # 滚动阻力系数
            rho = 1.225  # 空气密度
            g = 9.81  # 重力加速度
            
            # 创建一个临时DataFrame，只包含有速度的行
            speed_df = df[has_speed].copy()
            
            # 计算各种阻力
            air_resistance = 0.5 * CdA * rho * speed_df['speed']**3
            
            # 滚动阻力
            rolling_resistance = Crr * weight_kg * g * speed_df['speed']
            
            # 爬坡阻力 (如果有坡度数据)
            if 'slope' in speed_df.columns:
                # 转换百分比坡度为角度
                slope_rad = np.arctan(speed_df['slope'].fillna(0) / 100)
                climbing_resistance = weight_kg * g * np.sin(slope_rad) * speed_df['speed']
            else:
                climbing_resistance = 0
            
            # 总功率估计
            physics_power = air_resistance + rolling_resistance + climbing_resistance
            
            # 效率因子 (考虑人体效率约20-25%)
            efficiency = 0.23
            physics_power = physics_power / efficiency
            
            # 添加基础代谢功率
            physics_power += weight_kg * 0.5
            
            # 确保物理功率模型不产生负值
            physics_power = physics_power.clip(lower=weight_kg * 0.5)
            
            # 设置合理的上限
            max_sustainable_power = weight_kg * 5  # ~5W/kg是业余骑行者可持续的上限
            
            # 如果预测功率超过物理模型的两倍或超过合理上限，进行调整
            unreasonable = (speed_df['predicted_power'] > physics_power * 2) | \
                          (speed_df['predicted_power'] > max_sustainable_power)
            
            if unreasonable.any():
                # 获取需要调整的行
                adjust_rows = speed_df[unreasonable].index
                
                # 将不合理值替换为物理模型与原预测的加权平均
                original_power = speed_df.loc[unreasonable, 'predicted_power']
                adjusted_power = physics_power[unreasonable] * 0.7 + original_power * 0.3
                
                # 确保调整后的功率不为负
                adjusted_power = adjusted_power.clip(lower=0)
                
                # 使用正确的索引更新原始DataFrame
                df.loc[adjust_rows, 'predicted_power'] = adjusted_power
                print(f"已调整 {len(adjust_rows)} 行不合理的功率预测")
    
    # 确保函数返回的数据不包含负功率值
    if 'predicted_power' in df.columns:
        neg_count = (df['predicted_power'] < 0).sum()
        if neg_count > 0:
            print(f"警告: 物理约束计算生成了 {neg_count} 个负值功率，已修正为0")
            df['predicted_power'] = df['predicted_power'].clip(lower=0)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='使用XGBoost模型预测骑行功率')
    parser.add_argument('--model', type=str, default='model/power_prediction_model.pkl',
                       help='模型文件路径')
    parser.add_argument('--input', type=str, required=True,
                       help='CSV文件或包含CSV文件的目录')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理目录中的所有CSV文件')
    parser.add_argument('--rider-info', type=str, default='source-data/rider/rider_info.json',
                       help='骑行者信息JSON文件路径')
    
    args = parser.parse_args()
    
    # 加载模型
    model, features, transform_power = load_model(args.model)
    
    # 进行预测
    if args.batch:
        batch_predict(model, features, args.input, args.rider_info, transform_power)
    else:
        predict_from_csv(model, features, args.input, transform_power=transform_power)

if __name__ == "__main__":
    main()