import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import matplotlib as mpl
import json
from sklearn.impute import SimpleImputer

# 配置matplotlib使用中文字体
try:
    # 尝试配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    print("警告: 无法配置中文字体，图表中的中文可能无法正确显示")

def extract_rider_id(filename):
    """
    从文件名中提取骑行者ID
    格式：rider前缀到第一个下划线之间的部分
    例如：rider1_xxx.csv -> rider1
    """
    if filename.startswith('rider'):
        parts = filename.split('_', 1)
        if len(parts) > 0:
            return parts[0]
    return None

def load_csv_data(csv_dir):
    """
    从目录加载所有CSV文件并合并
    """
    dataframes = []
    
    for root, _, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                try:
                    df = pd.read_csv(filepath)
                    # 从文件名中提取骑行者ID并添加到DataFrame中
                    rider_id = extract_rider_id(file)
                    if rider_id:
                        df['rider_id'] = rider_id
                    dataframes.append(df)
                except Exception as e:
                    print(f"加载 {filepath} 时出错: {e}")
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return None

def preprocess_data(df, fill_method='interpolate', max_gap=None, window_size=5, column_specific_methods=None, detect_outliers=False, outlier_threshold=3.0):
    """
    对数据进行预处理，包括缺失值填充和异常值处理
    
    参数:
    df: 骑行数据DataFrame
    fill_method: 填充方法，可选值为:
        - 'interpolate': 线性插值填充
        - 'ffill': 前值填充
        - 'bfill': 后值填充
        - 'mean': 均值填充
        - 'median': 中位数填充
        - 'rolling_mean': 滑动窗口均值填充
        - 'rolling_median': 滑动窗口中位数填充
        - 'time_weighted': 基于时间权重的填充(适用于时间序列)
        - 'spline': 样条插值(适用于曲线数据)
        - 'polynomial': 多项式插值(适用于曲线数据)
        - 'knn': 基于K近邻的填充
        - 'imputer': 使用SimpleImputer进行填充
    max_gap: 最大填充间隔，仅对'interpolate'等插值方法有效，如果为None则不限制
    window_size: 滑动窗口大小，用于'rolling_mean'和'rolling_median'方法
    column_specific_methods: 字典，指定特定列使用的填充方法，格式为{列名: 填充方法}
    detect_outliers: 是否检测并处理异常值
    outlier_threshold: 异常值检测的阈值(标准差的倍数)
    
    返回:
    处理后的DataFrame
    """
    print(f"数据预处理前的行数: {len(df)}")
    print(f"默认填充方法: {fill_method}")
    
    # 检查缺失值情况
    na_count = df.isna().sum()
    if na_count.sum() > 0:
        print("\n检测到缺失值:")
        for col in na_count[na_count > 0].index:
            print(f"  {col}: {na_count[col]} 个缺失值 ({na_count[col]/len(df)*100:.2f}%)")
    else:
        print("\n未检测到缺失值")
        # 如果没有缺失值但需要检测异常值
        if detect_outliers:
            return handle_outliers(df, outlier_threshold)
        return df
    
    # 备份原始数据
    df_original = df.copy()
    
    # 处理列特定的填充方法
    if column_specific_methods is not None:
        print("\n使用列特定的填充方法:")
        for col, method in column_specific_methods.items():
            if col in df.columns and df[col].isna().any():
                print(f"  {col}: 使用 {method} 方法填充")
                df = fill_column(df, col, method, window_size, max_gap)
    
    # 对其余列使用默认填充方法
    remaining_cols = [col for col in df.columns if df[col].isna().any() and 
                     (column_specific_methods is None or col not in column_specific_methods)]
    
    if remaining_cols:
        print(f"\n对其余列使用默认填充方法 '{fill_method}':")
        for col in remaining_cols:
            df = fill_column(df, col, fill_method, window_size, max_gap)
    
    # 检查填充后的缺失值情况
    na_count_after = df.isna().sum()
    if na_count_after.sum() > 0:
        print("\n填充后仍有缺失值:")
        for col in na_count_after[na_count_after > 0].index:
            print(f"  {col}: {na_count_after[col]} 个缺失值")
        
        # 对于仍然有缺失值的列，使用0填充（仅对数值列）
        for col in na_count_after[na_count_after > 0].index:
            if np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(0, inplace=True)
                print(f"  {col}: 使用0填充剩余缺失值")
    else:
        print("\n所有缺失值已填充")
    
    # 输出填充前后的数据对比
    filled_count = na_count.sum() - na_count_after.sum()
    print(f"\n共填充了 {filled_count} 个缺失值")
    
    # 处理异常值
    if detect_outliers:
        df = handle_outliers(df, outlier_threshold)
    
    return df

def fill_column(df, column, method='linear', window_size=5, max_gap=None):
    """
    使用指定方法填充列中的缺失值
    
    参数:
    df: 数据框
    column: 要填充的列名
    method: 填充方法 ('linear', 'forward', 'rolling_mean', 'rolling_median')
    window_size: 滚动窗口大小（用于rolling方法）
    max_gap: 最大填充间隔（超过此间隔的值保持为NaN）
    
    返回:
    填充后的数据框
    """
    # 复制数据框以避免修改原始数据
    df = df.copy()
    
    # 如果列不存在或没有缺失值，直接返回
    if column not in df.columns or not df[column].isna().any():
        return df
    
    # 记录填充前的缺失值数量
    missing_before = df[column].isna().sum()
    
    # 应用指定的填充方法
    if method == 'linear':
        # 线性插值
        if max_gap:
            # 对于小于max_gap的间隔使用线性插值
            mask = df[column].isna()
            df[column] = df[column].interpolate(method='linear', limit=max_gap)
        else:
            df[column] = df[column].interpolate(method='linear')
    
    elif method == 'forward':
        # 前向填充
        if max_gap:
            df[column] = df[column].fillna(method='ffill', limit=max_gap)
        else:
            df[column] = df[column].fillna(method='ffill')
    
    elif method == 'rolling_mean':
        # 滚动平均填充
        filled_values = df[column].rolling(window=window_size, min_periods=1, center=True).mean()
        if max_gap:
            # 仅填充小于max_gap的间隔
            gaps = df[column].isna().astype(int).groupby(df[column].notna().astype(int).cumsum()).cumsum()
            mask = df[column].isna() & (gaps <= max_gap)
            df.loc[mask, column] = filled_values[mask]
        else:
            df.loc[df[column].isna(), column] = filled_values[df[column].isna()]
    
    elif method == 'rolling_median':
        # 滚动中位数填充
        filled_values = df[column].rolling(window=window_size, min_periods=1, center=True).median()
        if max_gap:
            # 仅填充小于max_gap的间隔
            gaps = df[column].isna().astype(int).groupby(df[column].notna().astype(int).cumsum()).cumsum()
            mask = df[column].isna() & (gaps <= max_gap)
            df.loc[mask, column] = filled_values[mask]
        else:
            df.loc[df[column].isna(), column] = filled_values[df[column].isna()]
    
    # 记录填充后的缺失值数量
    missing_after = df[column].isna().sum()
    filled_count = missing_before - missing_after
    
    if filled_count > 0:
        print(f"  {column}: 已填充 {filled_count} 个缺失值 (方法: {method})")
    
    return df

def handle_outliers(df, threshold=3.0):
    """
    检测并处理异常值
    
    参数:
    df: 数据DataFrame
    threshold: 异常值检测的阈值(标准差的倍数)
    
    返回:
    处理后的DataFrame
    """
    print(f"\n检测异常值 (阈值: {threshold} 倍标准差)")
    df_clean = df.copy()
    
    # 只处理数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    outliers_detected = False
    for col in numeric_cols:
        # 计算Z分数
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:  # 避免除以零
            continue
            
        z_scores = np.abs((df[col] - mean) / std)
        
        # 检测异常值
        outliers = z_scores > threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outliers_detected = True
            outlier_percent = outlier_count / len(df) * 100
            print(f"  {col}: 检测到 {outlier_count} 个异常值 ({outlier_percent:.2f}%)")
            
            # 替换异常值为列的中位数
            median_val = df[col].median()
            df_clean.loc[outliers, col] = median_val
    
    if not outliers_detected:
        print("  未检测到异常值")
    
    return df_clean

def prepare_features(df, rider_info=None, fill_method='interpolate', max_gap=None, column_specific_methods=None, detect_outliers=False, outlier_threshold=3.0, transform_power=False):
    """
    准备特征和目标变量，包括骑行者身高体重信息
    
    参数:
    df: 骑行数据DataFrame
    rider_info: 包含骑行者信息的字典，如 {'height': 175, 'weight': 70}
    fill_method: 填充方法，可选值参见preprocess_data函数的文档
    max_gap: 最大填充间隔，仅对'interpolate'等插值方法有效，如果为None则不限制
    column_specific_methods: 字典，指定特定列使用的填充方法，格式为{列名: 填充方法}
    detect_outliers: 是否检测并处理异常值
    outlier_threshold: 异常值检测的阈值(标准差的倍数)
    transform_power: 是否对功率进行对数变换
    """
    # 确保数据包含所需的列
    required_cols = ['power']
    
    # 可能的特征列表 - 移除altitude，添加latitude和longitude
    feature_cols = ['speed', 'heart_rate', 'cadence', 'distance', 'time_diff', 
                   'slope', 'temperature', 'latitude', 'longitude', 
                   'acceleration']
    
    # 添加身高体重信息（如果提供）
    if rider_info is not None:
        if 'height' in rider_info:
            df['height'] = rider_info['height']  # 添加身高列（厘米）
            feature_cols.append('height')
        
        if 'weight' in rider_info:
            df['weight'] = rider_info['weight']  # 添加体重列（公斤）
            feature_cols.append('weight')
            
            # 可以添加功率/体重比作为额外特征
            if 'power' in df.columns:
                df['power_to_weight'] = df['power'] / rider_info['weight']
    
    # 检查目标变量是否存在
    if 'power' not in df.columns:
        raise ValueError("CSV数据中没有'power'列")
    
    # 筛选出可用的特征
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        raise ValueError("未找到任何可用特征")
    
    # 确保latitude和longitude在特征中
    if 'latitude' not in available_features or 'longitude' not in available_features:
        print("警告: 数据中缺少latitude或longitude特征")
    
    # 对数据进行预处理和缺失值填充
    df = preprocess_data(df, fill_method=fill_method, max_gap=max_gap, 
                        column_specific_methods=column_specific_methods, 
                        detect_outliers=detect_outliers, 
                        outlier_threshold=outlier_threshold)
    
    # 确保功率值非负
    if 'power' in df.columns:
        neg_power_count = (df['power'] < 0).sum()
        if neg_power_count > 0:
            print(f"警告: 训练数据中存在 {neg_power_count} 个负值功率，已设置为0")
            df['power'] = df['power'].clip(lower=0)
    
    # 筛选出有效的骑行数据，在训练数据中排除静止状态
    # 这确保模型学习活动状态下的功率关系
    if 'speed' in df.columns:
        active_rows = df['speed'] > 0.5  # 定义有效速度阈值
        inactive_count = (~active_rows).sum()
        if inactive_count > 0:
            print(f"排除 {inactive_count} 行静止/低速数据（速度 <= 0.5 m/s）")
            df = df[active_rows].copy()
    
    # 如果有speed和cadence，添加speed_cadence交互特征
    if 'speed' in df.columns and 'cadence' in df.columns:
        df['speed_cadence'] = df['speed'] * df['cadence']
        feature_cols.append('speed_cadence')
        print("添加速度-踏频交互特征")
    
    # 添加心率的平方作为特征，捕捉非线性关系
    if 'heart_rate' in df.columns:
        df['heart_rate_squared'] = df['heart_rate'] ** 2
        feature_cols.append('heart_rate_squared')
        print("添加心率平方特征")
    
    # 添加速度的立方项，以更好地捕捉高速度时的功率关系
    if 'speed' in df.columns:
        df['speed_cubed'] = df['speed'] ** 3
        feature_cols.append('speed_cubed')
        print("添加速度立方特征")
    
    # 添加特征组合：心率与踏频的组合特征
    if 'heart_rate' in df.columns and 'cadence' in df.columns:
        df['hr_cadence'] = df['heart_rate'] * df['cadence']
        feature_cols.append('hr_cadence')
        print("添加心率-踏频组合特征")
    
    # 添加特征标志：标记数据质量
    # 创建数据质量指标列，表示行数据的完整性
    quality_score = pd.Series(1.0, index=df.index)
    for col in ['speed', 'heart_rate', 'cadence']:
        if col in df.columns:
            # 有缺失值的行质量分数降低
            quality_score[df[col].isna()] *= 0.7
    
    df['data_quality'] = quality_score
    feature_cols.append('data_quality')
    print("添加数据质量指标特征")
    
    # 对功率进行对数变换（可选，默认关闭）
    if transform_power and 'power' in df.columns:
        # 添加小常数避免log(0)
        df['power_original'] = df['power']
        df['power'] = np.log1p(df['power'])
        print("已对功率进行对数变换 (log(1+x))")
    
    # 移除异常值 - 调整四分位距，减少对高功率值的过滤
    for col in available_features:
        if col in df.columns:
            df = df[df[col].between(df[col].quantile(0.01), df[col].quantile(0.995))]  # 扩大上限到99.5%
    
    # 对功率值应用更宽松的过滤
    if 'power' in df.columns:
        df = df[df['power'].between(df['power'].quantile(0.01), df['power'].quantile(0.999))]  # 几乎保留所有高功率值
    
    # 准备特征和目标
    X = df[available_features]
    y = df['power']
    
    print(f"使用的特征: {available_features}")
    print(f"数据样本数: {len(df)}")
    
    return X, y, available_features, transform_power

def train_xgboost_model(X, y, eval_size=0.2, random_state=42):
    """
    训练XGBoost模型
    """
    # 划分训练集和验证集
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=eval_size, random_state=random_state
    )
    
    # XGBoost参数 - 调整以减少负值预测并提高高功率区域的预测准确度
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,        # 增加学习率
        'max_depth': 8,              # 增加树深度
        'min_child_weight': 2,       # 减小以允许更细的节点分裂
        'subsample': 0.85,           # 略微提高采样率
        'colsample_bytree': 0.85,    # 略微提高特征采样率
        'n_estimators': 1000,        # 增加树的数量
        'gamma': 0.05,               # 减小正则化
        'reg_alpha': 0.05,           # 减小L1正则化
        'reg_lambda': 0.5            # 减小L2正则化
    }
    
    # 训练模型
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_eval)
    
    # 检查负值预测
    neg_pred_count = (y_pred < 0).sum()
    if neg_pred_count > 0:
        print(f"警告: 验证集上有 {neg_pred_count} 个负值功率预测 ({neg_pred_count/len(y_pred)*100:.2f}%)")
    
    # 计算平均绝对误差和均方误差
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    r2 = r2_score(y_eval, y_pred)
    mae = np.mean(np.abs(y_eval - y_pred))
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    # 分析功率值高于平均值的样本的预测性能
    high_power_mask = y_eval > y_eval.mean()
    if high_power_mask.sum() > 0:
        high_power_rmse = np.sqrt(mean_squared_error(y_eval[high_power_mask], y_pred[high_power_mask]))
        high_power_r2 = r2_score(y_eval[high_power_mask], y_pred[high_power_mask])
        print(f"高功率区域RMSE: {high_power_rmse:.2f}")
        print(f"高功率区域R²: {high_power_r2:.2f}")
    
    return model, rmse, r2

def save_model(model, feature_names, model_path='model/power_prediction_model.pkl', transform_power=False):
    """
    保存模型和特征名称
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型和特征名称
    model_info = {
        'model': model,
        'features': feature_names,
        'transform_power': transform_power  # 记录是否进行了功率变换
    }
    
    joblib.dump(model_info, model_path)
    print(f"模型保存到: {model_path}")

def load_rider_info(rider_id, rider_info_path='source-data/rider/rider_info.json'):
    """
    从rider_info.json文件中获取骑行者信息
    """
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
    
    # 如果无法加载骑行者信息，返回None
    return None

def main():
    # 加载训练数据
    train_data = load_csv_data('source-data/train')
    if train_data is None:
        print("未找到训练数据")
        return
    
    # 获取数据中的骑行者ID
    if 'rider_id' in train_data.columns:
        # 使用数据中最常见的骑行者ID
        most_common_rider_id = train_data['rider_id'].mode()[0]
        print(f"使用最常见的骑行者ID: {most_common_rider_id}")
        
        # 获取对应的骑行者信息
        rider_info = load_rider_info(most_common_rider_id)
    else:
        print("警告: 数据中没有骑行者ID信息，使用默认值rider1")
        rider_info = load_rider_info('rider1')
    
    # 设置列特定的填充方法
    column_specific_methods = {
        'heart_rate': 'rolling_median',  # 心率使用滑动窗口中位数填充
        'power': 'rolling_mean',        # 功率使用滑动窗口均值填充
        'cadence': 'rolling_mean',      # 踏频使用滑动窗口均值填充
        'speed': 'spline',              # 速度使用样条插值
        'slope': 'polynomial',          # 坡度使用多项式插值
        'distance': 'interpolate',      # 距离使用线性插值
        'temperature': 'ffill'          # 温度使用前值填充
    }
    
    # 准备特征，使用增强的数据预处理
    X, y, features, transform_power = prepare_features(
        train_data, 
        rider_info=rider_info,
        fill_method='interpolate',
        max_gap=10,
        column_specific_methods=column_specific_methods,
        detect_outliers=True,
        outlier_threshold=3.0,
        transform_power=False  # 禁用功率对数变换
    )
    
    # 训练模型
    model, rmse, r2 = train_xgboost_model(X, y)
    
    # 保存模型
    save_model(model, features, transform_power=transform_power)
    
    # 验证集评估
    verify_data = load_csv_data('source-data/verify')
    if verify_data is not None:
        # 如果验证集有骑行者ID，使用相同的处理方式
        if 'rider_id' in verify_data.columns:
            most_common_rider_id = verify_data['rider_id'].mode()[0]
            print(f"验证集使用最常见的骑行者ID: {most_common_rider_id}")
            verify_rider_info = load_rider_info(most_common_rider_id)
        else:
            verify_rider_info = rider_info
        
        # 对验证集使用相同的预处理策略 - 修复这里的解包问题
        X_verify, y_verify, _, _ = prepare_features(  # 修改这行，添加一个变量接收transform_power
            verify_data, 
            rider_info=verify_rider_info,
            fill_method='interpolate',
            max_gap=10,
            column_specific_methods=column_specific_methods,
            detect_outliers=True,
            outlier_threshold=3.0,
            transform_power=transform_power
        )
        # 确保验证集有相同的特征
        missing_features = [f for f in features if f not in X_verify.columns]
        if missing_features:
            print(f"验证集缺少特征: {missing_features}")
            for feature in missing_features:
                X_verify[feature] = 0  # 用0填充缺失特征
        
        X_verify = X_verify[features]  # 保证特征顺序一致
        
        # 在验证集上评估
        y_pred = model.predict(X_verify)
        verify_rmse = np.sqrt(mean_squared_error(y_verify, y_pred))
        verify_r2 = r2_score(y_verify, y_pred)
        
        print(f"验证集RMSE: {verify_rmse:.2f}")
        print(f"验证集R²: {verify_r2:.2f}")
        
        # 绘制实际值与预测值对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_verify, y_pred, alpha=0.5)
        plt.plot([min(y_verify), max(y_verify)], [min(y_verify), max(y_verify)], 'r--')
        plt.xlabel('实际功率')
        plt.ylabel('预测功率')
        plt.title('功率预测对比')
        plt.tight_layout()
        plt.savefig('view/prediction_comparison.png')

if __name__ == "__main__":
    main()