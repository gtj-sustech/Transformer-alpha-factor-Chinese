"""
量化Transformer项目 - 阶段1：数据准备
环境：M1 iMac, VSCode, PyTorch, Akshare
目标：从原始数据生成可用于模型训练的特征DataFrame和标签Series
"""
import akshare as ak
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import pickle
import os
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class QuantDataPreprocessor:
    """量化数据预处理器"""
    
    def __init__(self, start_date='20200101', end_date='20251231', lookback_days=30, forward_days=5):
        """
        初始化预处理器
        
        参数:
            start_date: 数据开始日期，格式'YYYYMMDD'
            end_date: 数据结束日期，格式'YYYYMMDD'
            lookback_days: 模型回看天数（序列长度）
            forward_days: 预测未来天数（标签周期）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.forward_days = forward_days
        
        # 沪深300成分股作为股票池（可替换）
        self.stock_pool = self.get_hs300_constituents()  # 这里调用的是 get_hs300_constituents
        print(f"获取到 {len(self.stock_pool)} 只股票")
    
    def get_hs300_constituents(self):
        """获取沪深300成分股列表 - 简化版本"""
        # 直接返回一些常见的沪深300成分股（这里只列出一部分用于测试）
        hs300_stocks = [
            '000001', '000002', '000063', '000066', '000069', '000100', '000157', 
            '000166', '000333', '000338', '000425', '000538', '000568', '000596',
            '000625', '000627', '000651', '000656', '000661', '000703', '000708',
            '000709', '000723', '000725', '000728', '000768', '000776', '000783',
            '000858', '000876', '000895', '000938', '000961', '000963', '001979',
            '002007', '002008', '002024', '002027', '002032', '002044', '002050',
            '002120', '002129', '002142', '002153', '002179', '002230', '002236',
            '002241', '002304', '002311', '002352', '002410', '002415', '002422',
            '002456', '002460', '002466', '002475', '002493', '002594', '002601',
            '002602', '002607', '002624', '002673', '002714', '002736', '002739',
            '002773', '002812', '002821', '002831', '002841', '002916', '002938',
            '002939', '002945', '002958', '003816', '300003', '300014', '300015',
            '300033', '300059', '300122', '300124', '300142', '300144', '300146',
            '300347', '300408', '300413', '300433', '300450', '300454', '300496',
            '300498', '300529', '300558', '300595', '300601', '300628', '300661',
            '300676', '300725', '300750', '300751', '300760', '300763', '300782',
            '300832', '300869', '300896', '300919', '300957', '300979', '300999',
            '600000', '600004', '600009', '600010', '600011', '600015', '600016',
            '600018', '600019', '600025', '600028', '600029', '600030', '600031',
            '600036', '600038', '600048', '600050', '600061', '600066', '600068',
            '600085', '600089', '600104', '600109', '600111', '600115', '600118',
            '600150', '600176', '600177', '600183', '600188', '600196', '600208',
            '600219', '600221', '600233', '600276', '600297', '600299', '600309',
            '600332', '600340', '600346', '600362', '600369', '600372', '600383',
            '600390', '600398', '600406', '600436', '600438', '600482', '600487',
            '600489', '600498', '600516', '600519', '600522', '600547', '600570',
            '600584', '600585', '600588', '600606', '600637', '600660', '600663',
            '600674', '600690', '600703', '600705', '600741', '600745', '600760',
            '600763', '600795', '600809', '600837', '600848', '600867', '600886',
            '600887', '600893', '600900', '600905', '600918', '600919', '600926',
            '600928', '600929', '600933', '600936', '600938', '600941', '600956',
            '600958', '600989', '600999', '601006', '601009', '601012', '601018',
            '601021', '601038', '601058', '601066', '601077', '601088', '601099',
            '601108', '601111', '601117', '601138', '601155', '601166', '601169',
            '601186', '601198', '601211', '601216', '601225', '601229', '601231',
            '601236', '601238', '601288', '601318', '601319', '601328', '601336',
            '601360', '601377', '601390', '601398', '601555', '601577', '601600',
            '601601', '601607', '601618', '601628', '601633', '601658', '601665',
            '601668', '601669', '601688', '601698', '601699', '601727', '601766',
            '601788', '601800', '601808', '601816', '601818', '601828', '601838',
            '601857', '601865', '601866', '601868', '601872', '601877', '601878',
            '601881', '601888', '601898', '601899', '601901', '601916', '601919',
            '601933', '601939', '601956', '601958', '601966', '601985', '601988',
            '601989', '601992', '601995', '601998', '603019', '603160', '603259',
            '603288', '603290', '603369', '603501', '603508', '603515', '603520',
            '603583', '603605', '603613', '603658', '603659', '603799', '603806',
            '603816', '603833', '603858', '603882', '603899', '603986', '603993',
            '605499', '688008', '688009', '688012', '688036', '688111', '688116',
            '688126', '688169', '688180', '688185', '688187', '688223', '688256',
            '688363', '688396', '688561', '688599', '688777', '688981'
        ]
        
        print(f"使用内置沪深300成分股列表，共{len(hs300_stocks)}只股票")
        return hs300_stocks[:300]  
    
    def fetch_stock_data(self, stock_code, retry=3):
        """
        获取单只股票的日K线数据
        
        参数:
            stock_code: 股票代码，支持多种格式
        """
        for i in range(retry):
            try:
                # 尝试不同的代码格式
                code_formats = [
                    stock_code,  # 原始格式
                    f"sh{stock_code}" if stock_code.startswith('6') else f"sz{stock_code}",  # 市场前缀
                    stock_code.zfill(6),  # 补零到6位
                    f"{stock_code}.SH" if stock_code.startswith('6') else f"{stock_code}.SZ",  # 交易所后缀
                ]
                
                for code_format in code_formats:
                    try:
                        df = ak.stock_zh_a_hist(symbol=code_format, period="daily", 
                                                start_date=self.start_date, 
                                                end_date=self.end_date, adjust="qfq")
                        
                        if df is not None and len(df) > 0:
                            df['股票代码'] = stock_code
                            df.rename(columns={
                                '日期': 'date',
                                '开盘': 'open',
                                '收盘': 'close',
                                '最高': 'high',
                                '最低': 'low',
                                '成交量': 'volume',
                                '成交额': 'amount',
                                '振幅': 'amplitude',
                                '涨跌幅': 'pct_change',
                                '涨跌额': 'change',
                                '换手率': 'turnover'
                            }, inplace=True)
                            
                            # 只保留需要的列
                            needed_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', '股票代码']
                            available_columns = [col for col in needed_columns if col in df.columns]
                            df = df[available_columns]
                            
                            df['date'] = pd.to_datetime(df['date'])
                            df.sort_values('date', inplace=True)
                            df.reset_index(drop=True, inplace=True)
                            
                            print(f"成功获取股票 {stock_code} 数据，共 {len(df)} 行")
                            return df
                            
                    except Exception as e:
                        continue
                
            except Exception as e:
                if i == retry - 1:
                    print(f"股票 {stock_code} 数据获取失败，已尝试 {retry} 次: {e}")
                continue
        
        return None
    
    def calculate_features(self, df):
        """计算技术指标特征"""
        if df is None or len(df) < 30:
            return None
            
        # 创建特征副本
        features = df.copy()
        
        # 基础价格特征
        features['returns'] = features['close'].pct_change()
        features['high_low_pct'] = (features['high'] - features['low']) / features['low']
        features['open_close_pct'] = (features['close'] - features['open']) / features['open']
        
        # 成交量特征，rollong(5).mean()表示移动计算过去5天的均值
        features['volume_ma5'] = features['volume'].rolling(5).mean()
        features['volume_ma20'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma20'] #衡量成交量是否异常放大或缩小
        
        # 价格移动平均线
        features['ma5'] = features['close'].rolling(5).mean()
        features['ma10'] = features['close'].rolling(10).mean()
        features['ma20'] = features['close'].rolling(20).mean()
        features['ma60'] = features['close'].rolling(60).mean()
        
        # 价格与均线关系
        features['close_vs_ma5'] = features['close'] / features['ma5'] - 1
        features['close_vs_ma20'] = features['close'] / features['ma20'] - 1
        features['ma5_vs_ma20'] = features['ma5'] / features['ma20'] - 1
        
        # 波动率特征，rolling.(5).std()表示滚动计算过去5日变量的标准差
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # 动量特征 公式：当天价格/N天前价格 -1，表示N天内的涨跌幅。shift(N)表示dataframe向下移动N行，最上面N行会变成NaN
        features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
        features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
        features['momentum_20'] = features['close'] / features['close'].shift(20) - 1
        
        # RSI相对强弱指数 (简化版)，diff()计算当前行与前一行的差值
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带 (简化版)
        features['bb_middle'] = features['close'].rolling(20).mean()
        bb_std = features['close'].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * bb_std
        features['bb_lower'] = features['bb_middle'] - 2 * bb_std
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']) #价格在布林带中的位置
        
        # 删除因滚动计算产生的NaN行
        features = features.dropna()
        
        return features
   
    def create_labels(self, df):
        """创建标签：未来N日收益率"""
        if df is None or len(df) < self.forward_days + 5:
            return None
            
        # 计算未来N日的收益率
        # 注意: 这里使用shift(-forward_days)是因为我们想用当前数据预测未来。
        # 将后forward_days天的收盘价移动到当前行就是df['future_close']
        df = df.copy()
        df['future_close'] = df['close'].shift(-self.forward_days)
        df['future_return'] = df['future_close'] / df['close'] - 1
        
        # 删除最后forward_days行（没有未来价格）,因为最后这些行由于shift操作向上移动会产生NaN
        df = df.iloc[:-self.forward_days] if self.forward_days > 0 else df
        
        return df[['date', '股票代码', 'future_return']]
    
    def prepare_all_data(self, max_stocks=None):
        """准备所有股票的数据"""
        all_features = []
        all_labels = []
        all_prices = []  # ✅ 新增：用于保存原始收盘价数据
        
        # 限制股票数量以便测试，如果max_stocks为None则处理所有股票
        stock_list = self.stock_pool[:max_stocks] if max_stocks else self.stock_pool
        
        print("开始获取并处理股票数据...")
        for i, stock_code in enumerate(stock_list):
            if (i + 1) % 10 == 0:
                print(f"处理进度: {i + 1}/{len(stock_list)}")
            
            # 1. 获取原始数据
            raw_data = self.fetch_stock_data(stock_code)
            if raw_data is None or len(raw_data) < 60:  # 至少需要60天数据
                continue
            
            # ✅ 新增：提取原始收盘价数据
            # 在计算特征之前保存原始收盘价
            if 'close' in raw_data.columns:
                price_subset = raw_data[['date', '股票代码', 'close']].copy()
                # 重命名为更通用的'price'，避免与特征中的'close'列混淆
                price_subset.rename(columns={'close': 'price'}, inplace=True)
                all_prices.append(price_subset)
                # 可选：打印调试信息
                if len(all_prices) <= 3:  # 只打印前3只股票作为示例
                    print(f"  股票 {stock_code}: 提取到 {len(price_subset)} 条价格数据")
            
            # 2. 计算特征
            feature_data = self.calculate_features(raw_data)
            if feature_data is None:
                continue
                
            # 3. 创建标签
            label_data = self.create_labels(feature_data)
            if label_data is None:
                continue
            
            # 4. 提取特征列（排除原始价格和成交量）
            feature_columns = [
                'returns', 'high_low_pct', 'open_close_pct',
                'volume_ma5', 'volume_ma20', 'volume_ratio',
                'ma5', 'ma10', 'ma20', 'ma60',
                'close_vs_ma5', 'close_vs_ma20', 'ma5_vs_ma20',
                'volatility_5', 'volatility_20',
                'momentum_5', 'momentum_10', 'momentum_20',
                'rsi', 'bb_position'
            ]
            
            # 确保所有特征列都存在
            available_features = [col for col in feature_columns if col in feature_data.columns]
            features_subset = feature_data[['date', '股票代码'] + available_features].copy()
            
            # 5. 合并特征和标签
            merged_data = pd.merge(features_subset, label_data, on=['date', '股票代码'], how='inner')
            
            if len(merged_data) > self.lookback_days:
                all_features.append(features_subset)
                all_labels.append(label_data)
        
        print(f"成功处理 {len(all_features)} 只股票")
        
        if len(all_features) == 0:
            raise ValueError("没有成功获取任何股票数据，请检查网络连接或股票代码")
        
        # 6. 合并所有股票数据
        features_combined = pd.concat(all_features, ignore_index=True)
        labels_combined = pd.concat(all_labels, ignore_index=True)
        
        # ✅ 新增：合并并保存价格数据
        if all_prices:
            prices_combined = pd.concat(all_prices, ignore_index=True)
            
            # 创建数据透视表：日期为行索引，股票代码为列，值为价格
            prices_pivot = pd.pivot_table(
                prices_combined, 
                values='price', 
                index='date', 
                columns='股票代码'
            )
            
            # 确保日期是datetime类型并按日期排序
            prices_pivot.index = pd.to_datetime(prices_pivot.index)
            prices_pivot = prices_pivot.sort_index()
            
            # 保存价格数据到文件
            if not os.path.exists('./data'):
                os.makedirs('./data')
            
            prices_file = './data/transformer_quant_data_prices.pkl'
            with open(prices_file, 'wb') as f:
                pickle.dump(prices_pivot, f)
            
            print(f"价格数据已保存: {prices_file}")
            print(f"价格数据形状: {prices_pivot.shape}")
            print(f"价格数据日期范围: {prices_pivot.index.min()} 到 {prices_pivot.index.max()}")
        else:
            print("警告: 未能提取到任何价格数据")
        
        # 7. 数据清洗与对齐
        features_cleaned, labels_aligned = self.clean_and_align_data(features_combined, labels_combined)
        
        # 8. 特征标准化
        features_normalized = self.normalize_features(features_cleaned)
        
        return features_normalized, labels_aligned

    def clean_and_align_data(self, features_df, labels_df):
        """清洗数据并对齐特征和标签"""
        #因为某些股票的某些特征可能缺失，或者标签缺失，所以需要对齐

        # 1. 删除包含NaN的行
        features_cleaned = features_df.dropna()
        labels_cleaned = labels_df.dropna()
        
        # 2. 创建多索引 (date, 股票代码)，即.set_index(['date', '股票代码'])的效果是先按date列排序，再按股票代码排序
        # implace=True表示直接在原DataFrame上修改
        features_cleaned.set_index(['date', '股票代码'], inplace=True)
        labels_cleaned.set_index(['date', '股票代码'], inplace=True)
        
        # 3. 对齐索引（取交集），找到特征和标签都有的日期-股票组合，取两个索引的交集
        # 确保每个样本既有特征又有标签
        common_index = features_cleaned.index.intersection(labels_cleaned.index)
        features_aligned = features_cleaned.loc[common_index]
        labels_aligned = labels_cleaned.loc[common_index]
        
        print(f"对齐后数据形状: 特征={features_aligned.shape}, 标签={labels_aligned.shape}")
        
        return features_aligned, labels_aligned
    
    def normalize_features(self, features_df):
        """按时间滚动进行横截面标准化"""
        
        print("开始特征标准化...")
        
        # 重置索引以便按日期分组，就是恢复之前的0，1，2，3索引，
        # 把之前的date和股票代码从索引变成普通列
        features_reset = features_df.reset_index()
        
        # 按日期分组，对每个横截面进行标准化
        normalized_features = []
        
        # 获取所有唯一日期并排序，以确保按时间顺序处理，即date列中所有不重复的日期排序放置
        unique_dates = sorted(features_reset['date'].unique())
        
        #下面的for循环是对每个日期的数据进行处理，dataframe格式，之后再把处理好的数据放入normalized_features列表中
        for date in unique_dates:
            daily_data = features_reset[features_reset['date'] == date].copy()
            
            # 对每个特征列进行标准化
            for col in daily_data.columns:
                if col not in ['date', '股票代码'] and daily_data[col].std() > 0:
                    # 横截面标准化: (x - mean) / std
                    daily_data[col] = (daily_data[col] - daily_data[col].mean()) / daily_data[col].std()
            
            normalized_features.append(daily_data)

        # 合并所有日期的数据，normalized_features是一个列表，里面每个元素是一个dataframe
        # pd.concat函数把这些dataframe沿着行方向上下拼接起来
        features_normalized = pd.concat(normalized_features, ignore_index=True)
        
        # 重新设置多索引
        features_normalized.set_index(['date', '股票代码'], inplace=True)
        
        # 再次处理极端值（Winsorization），clip函数用于限制数值在指定范围内
        # 注意，这里是对整个数据集缩尾，不是按日期分别缩尾处理
        for col in features_normalized.columns:
            if features_normalized[col].std() > 0:
                # 将极端值限制在[-3, 3]个标准差内
                mean_val = features_normalized[col].mean()
                std_val = features_normalized[col].std()
                features_normalized[col] = features_normalized[col].clip(
                    mean_val - 3 * std_val, 
                    mean_val + 3 * std_val
                )
        
        print("特征标准化完成")
        return features_normalized
    
    def save_data(self, features_df, labels_series, file_prefix='quant_data'):
        """保存处理好的数据"""
        
        # 创建数据目录
        if not os.path.exists('./data'):
            os.makedirs('./data')
        
        # 保存为pickle文件
        features_file = f'./data/{file_prefix}_features.pkl'
        labels_file = f'./data/{file_prefix}_labels.pkl'
        
        # 确保labels是Series格式
        if isinstance(labels_series, pd.DataFrame):
            labels_series = labels_series['future_return']
        
        with open(features_file, 'wb') as f:
            pickle.dump(features_df, f)
        
        with open(labels_file, 'wb') as f:
            pickle.dump(labels_series, f)
        
        print(f"数据已保存至: {features_file}, {labels_file}")
        
        # ✅ 新增：尝试加载并保存价格数据
        try:
            prices_file = './data/transformer_quant_data_prices.pkl'
            if os.path.exists(prices_file):
                with open(prices_file, 'rb') as f:
                    prices_df = pickle.load(f)
                
                # 也保存为CSV以便查看
                prices_df.to_csv(f'./data/{file_prefix}_prices.csv', encoding='utf-8-sig')
                print(f"价格数据已保存: ./data/{file_prefix}_prices.csv")
        except Exception as e:
            print(f"保存价格数据时出错: {e}")
        
        # 也保存为CSV以便查看
        features_df.reset_index().to_csv(f'./data/{file_prefix}_features.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(labels_series).reset_index().to_csv(f'./data/{file_prefix}_labels.csv', index=False, encoding='utf-8-sig')
        
        return features_file, labels_file

def main():
    """主函数：执行数据准备流程 (严格时间划分版本)"""
    
    print("=" * 60)
    print("量化Transformer项目 - 数据准备阶段 (严格时间划分)")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 初始化预处理器 - 获取全量数据
    preprocessor = QuantDataPreprocessor(
        start_date='20240101',
        end_date='20251231',  
        lookback_days=30,
        forward_days=5
    )
    
    # 调试：打印股票池
    print(f"\n股票池大小: {len(preprocessor.stock_pool)}")
    print(f"前20只股票: {preprocessor.stock_pool[:20]}")
    
    # 测试获取单只股票数据
    test_stock = preprocessor.stock_pool[0]
    print(f"\n测试获取股票 {test_stock} 的数据...")
    test_data = preprocessor.fetch_stock_data(test_stock)
    if test_data is not None:
        print(f"测试成功，获取到 {len(test_data)} 行数据")
        print(test_data.head())
    else:
        print("测试失败，请检查akshare接口")
    
    try:
        print("\n开始获取和预处理全量数据...")
        # 获取全量数据（全部300只股票）
        features_df, labels_series = preprocessor.prepare_all_data(max_stocks=300)
        
        # 数据概览
        print("\n全量数据概览:")
        print(f"特征数据形状: {features_df.shape}")
        print(f"标签数据形状: {labels_series.shape}")
        
        print("\n特征列列表:")
        print(features_df.columns.tolist())
        
        print("\n数据完整性检查:")
        print(f"特征数据中NaN值数量: {features_df.isna().sum().sum()}")
        print(f"标签数据中NaN值数量: {pd.isna(labels_series).sum() if hasattr(labels_series, 'isna') else 'N/A'}")
        
        # ==================== 严格时间划分 ====================
        print("\n" + "=" * 60)
        print("开始严格时间划分数据集")
        print("=" * 60)
        
        # 获取所有唯一日期并排序
        all_dates = sorted(features_df.index.get_level_values(0).unique())
        total_days = len(all_dates)
        
        # 划分比例：70%训练，15%验证，15%测试（回测）
        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15
        
        train_end_idx = int(total_days * train_ratio)
        val_end_idx = train_end_idx + int(total_days * val_ratio)
        
        # 确保有足够的交易日
        train_end_idx = max(train_end_idx, 100)  # 至少100天训练
        val_end_idx = min(val_end_idx, total_days - 50)  # 至少保留50天测试
        
        train_end_date = all_dates[train_end_idx - 1]
        val_end_date = all_dates[val_end_idx - 1]
        test_start_date = all_dates[val_end_idx]
        
        print(f"\n=== 时间划分详情 ===")
        print(f"总交易日数: {total_days}")
        print(f"训练集: {all_dates[0].date()} 到 {train_end_date.date()} (共{train_end_idx}天)")
        print(f"验证集: {all_dates[train_end_idx].date()} 到 {val_end_date.date()} (共{val_end_idx - train_end_idx}天)")
        print(f"测试集(回测): {test_start_date.date()} 到 {all_dates[-1].date()} (共{total_days - val_end_idx}天)")
        
        # 定义按日期筛选的函数
        def filter_by_date(df, start_date, end_date):
            """筛选指定日期范围内的数据"""
            return df[(df.index.get_level_values('date') >= start_date) &
                      (df.index.get_level_values('date') <= end_date)]
        
        # 划分数据集
        train_features = filter_by_date(features_df, all_dates[0], train_end_date)
        train_labels = filter_by_date(labels_series, all_dates[0], train_end_date)
        
        val_features = filter_by_date(features_df, all_dates[train_end_idx], val_end_date)
        val_labels = filter_by_date(labels_series, all_dates[train_end_idx], val_end_date)
        
        test_features = filter_by_date(features_df, test_start_date, all_dates[-1])
        test_labels = filter_by_date(labels_series, test_start_date, all_dates[-1])
        
        # 检查各数据集大小
        print(f"\n=== 数据集大小检查 ===")
        print(f"训练集: {train_features.shape[0]} 个样本")
        print(f"验证集: {val_features.shape[0]} 个样本")
        print(f"测试集: {test_features.shape[0]} 个样本")
        
        # 检查是否有重叠日期
        train_dates = set(train_features.index.get_level_values(0))
        val_dates = set(val_features.index.get_level_values(0))
        test_dates = set(test_features.index.get_level_values(0))
        
        if train_dates.intersection(val_dates):
            print("警告: 训练集和验证集有日期重叠!")
        if train_dates.intersection(test_dates):
            print("警告: 训练集和测试集有日期重叠!")
        if val_dates.intersection(test_dates):
            print("警告: 验证集和测试集有日期重叠!")
        
        # ==================== 保存数据集 ====================
        print("\n" + "=" * 60)
        print("保存划分后的数据集")
        print("=" * 60)
        
        # 创建数据目录（如果不存在）
        if not os.path.exists('./data'):
            os.makedirs('./data')
        
        # 创建CSV保存目录
        csv_dir = './data/csv_exports'
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # 1. 保存pickle文件（用于训练）
        print("\n1. 保存pickle文件（用于训练）...")
        
        # 保存训练集
        train_features_file = './data/train_features.pkl'
        train_labels_file = './data/train_labels.pkl'
        with open(train_features_file, 'wb') as f:
            pickle.dump(train_features, f)
        with open(train_labels_file, 'wb') as f:
            pickle.dump(train_labels, f)
        
        # 保存验证集
        val_features_file = './data/val_features.pkl'
        val_labels_file = './data/val_labels.pkl'
        with open(val_features_file, 'wb') as f:
            pickle.dump(val_features, f)
        with open(val_labels_file, 'wb') as f:
            pickle.dump(val_labels, f)
        
        # 保存测试集
        test_features_file = './data/test_features.pkl'
        test_labels_file = './data/test_labels.pkl'
        with open(test_features_file, 'wb') as f:
            pickle.dump(test_features, f)
        with open(test_labels_file, 'wb') as f:
            pickle.dump(test_labels, f)
        
        # 2. 保存CSV文件（用于查看）
        print("\n2. 保存CSV文件（用于查看）...")
        
        # 保存训练集CSV（前1000行示例，避免文件过大）
        train_features_csv = f'{csv_dir}/train_features_sample.csv'
        train_labels_csv = f'{csv_dir}/train_labels_sample.csv'
        
        # 重置索引以便保存为CSV
        train_features_sample = train_features.reset_index().head(1000)
        train_labels_sample = train_labels.reset_index().head(1000)
        
        train_features_sample.to_csv(train_features_csv, index=False, encoding='utf-8-sig')
        train_labels_sample.to_csv(train_labels_csv, index=False, encoding='utf-8-sig')
        
        # 保存验证集CSV
        val_features_csv = f'{csv_dir}/val_features_sample.csv'
        val_labels_csv = f'{csv_dir}/val_labels_sample.csv'
        
        val_features_sample = val_features.reset_index().head(500)
        val_labels_sample = val_labels.reset_index().head(500)
        
        val_features_sample.to_csv(val_features_csv, index=False, encoding='utf-8-sig')
        val_labels_sample.to_csv(val_labels_csv, index=False, encoding='utf-8-sig')
        
        # 保存测试集CSV
        test_features_csv = f'{csv_dir}/test_features_sample.csv'
        test_labels_csv = f'{csv_dir}/test_labels_sample.csv'
        
        test_features_sample = test_features.reset_index().head(500)
        test_labels_sample = test_labels.reset_index().head(500)
        
        test_features_sample.to_csv(test_features_csv, index=False, encoding='utf-8-sig')
        test_labels_sample.to_csv(test_labels_csv, index=False, encoding='utf-8-sig')
        
        # 3. 保存价格数据
        print("\n3. 处理价格数据...")
        try:
            with open('./data/transformer_quant_data_prices.pkl', 'rb') as f:
                prices_df = pickle.load(f)
            
            # 划分价格数据
            train_prices = prices_df.loc[all_dates[0]:train_end_date]
            val_prices = prices_df.loc[all_dates[train_end_idx]:val_end_date]
            test_prices = prices_df.loc[test_start_date:all_dates[-1]]
            
            # 保存pickle文件
            train_prices.to_pickle('./data/train_prices.pkl')
            val_prices.to_pickle('./data/val_prices.pkl')
            test_prices.to_pickle('./data/test_prices.pkl')
            
            # 保存价格数据CSV（示例）
            train_prices_sample = train_prices.head(20).T.head(10)  # 前20天，前10只股票
            val_prices_sample = val_prices.head(20).T.head(10)
            test_prices_sample = test_prices.head(20).T.head(10)
            
            train_prices_sample.to_csv(f'{csv_dir}/train_prices_sample.csv', encoding='utf-8-sig')
            val_prices_sample.to_csv(f'{csv_dir}/val_prices_sample.csv', encoding='utf-8-sig')
            test_prices_sample.to_csv(f'{csv_dir}/test_prices_sample.csv', encoding='utf-8-sig')
            
            print("   价格数据划分并保存完成")
        except Exception as e:
            print(f"   价格数据处理时出错: {e}")
        
        # 4. 创建数据划分汇总报告
        print("\n4. 创建数据划分汇总报告...")
        summary_report = f"""
数据划分汇总报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、总体信息
总交易日数: {total_days}
总股票数量: {len(features_df.index.get_level_values('股票代码').unique())}
特征维度: {features_df.shape[1]}

二、时间划分详情
训练集时间: {all_dates[0].date()} 到 {train_end_date.date()} (共{train_end_idx}天)
验证集时间: {all_dates[train_end_idx].date()} 到 {val_end_date.date()} (共{val_end_idx - train_end_idx}天)
测试集时间: {test_start_date.date()} 到 {all_dates[-1].date()} (共{total_days - val_end_idx}天)

三、样本数量统计
训练集样本数: {train_features.shape[0]}
验证集样本数: {val_features.shape[0]}
测试集样本数: {test_features.shape[0]}
总计样本数: {train_features.shape[0] + val_features.shape[0] + test_features.shape[0]}

四、股票数量统计
训练集股票数: {len(train_features.index.get_level_values('股票代码').unique())}
验证集股票数: {len(val_features.index.get_level_values('股票代码').unique())}
测试集股票数: {len(test_features.index.get_level_values('股票代码').unique())}

五、文件保存位置
1. Pickle文件（用于训练）:
   - 训练集: ./data/train_features.pkl, ./data/train_labels.pkl
   - 验证集: ./data/val_features.pkl, ./data/val_labels.pkl
   - 测试集: ./data/test_features.pkl, ./data/test_labels.pkl

2. CSV示例文件（用于查看）:
   - 训练集: {csv_dir}/train_features_sample.csv, {csv_dir}/train_labels_sample.csv
   - 验证集: {csv_dir}/val_features_sample.csv, {csv_dir}/val_labels_sample.csv
   - 测试集: {csv_dir}/test_features_sample.csv, {csv_dir}/test_labels_sample.csv

六、特征列列表
{features_df.columns.tolist()}

七、数据完整性检查
训练集特征NaN数量: {train_features.isna().sum().sum()}
验证集特征NaN数量: {val_features.isna().sum().sum()}
测试集特征NaN数量: {test_features.isna().sum().sum()}

注意：CSV示例文件只保存了前1000/500行数据，完整数据请查看pickle文件。
"""
        
        # 保存汇总报告
        summary_file = f'{csv_dir}/data_split_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # 5. 创建数据预览文件
        print("\n5. 创建数据预览文件...")
        
        # 创建各数据集的统计信息
        def create_dataset_stats(features_df, name):
            stats = {
                '数据集': name,
                '开始日期': features_df.index.get_level_values(0).min().date(),
                '结束日期': features_df.index.get_level_values(0).max().date(),
                '样本数量': features_df.shape[0],
                '股票数量': len(features_df.index.get_level_values('股票代码').unique()),
                '特征数量': features_df.shape[1],
                'NaN数量': features_df.isna().sum().sum(),
                '日期数量': len(features_df.index.get_level_values(0).unique())
            }
            return stats
        
        # 收集所有数据集的统计信息
        datasets_stats = [
            create_dataset_stats(train_features, '训练集'),
            create_dataset_stats(val_features, '验证集'),
            create_dataset_stats(test_features, '测试集')
        ]
        
        # 转换为DataFrame并保存为CSV
        stats_df = pd.DataFrame(datasets_stats)
        stats_file = f'{csv_dir}/datasets_statistics.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        # 6. 创建每日样本数量统计
        print("\n6. 创建每日样本数量统计...")
        
        # 计算每个交易日的样本数量
        def count_samples_by_date(features_df, name):
            date_counts = features_df.index.get_level_values(0).value_counts().sort_index()
            df = pd.DataFrame({
                'date': date_counts.index,
                f'{name}_样本数': date_counts.values
            })
            return df
        
        # 获取各数据集的每日样本数
        train_date_counts = count_samples_by_date(train_features, '训练集')
        val_date_counts = count_samples_by_date(val_features, '验证集')
        test_date_counts = count_samples_by_date(test_features, '测试集')
        
        # 合并所有数据集的每日样本数
        all_date_counts = pd.DataFrame({'date': all_dates})
        all_date_counts = all_date_counts.merge(train_date_counts, on='date', how='left')
        all_date_counts = all_date_counts.merge(val_date_counts, on='date', how='left')
        all_date_counts = all_date_counts.merge(test_date_counts, on='date', how='left')
        all_date_counts = all_date_counts.fillna(0)
        
        # 保存每日样本数统计
        date_stats_file = f'{csv_dir}/daily_sample_counts.csv'
        all_date_counts.to_csv(date_stats_file, index=False, encoding='utf-8-sig')
        
        print(f"\n数据集保存完成!")
        print(f"Pickle文件已保存到 ./data/ 目录")
        print(f"CSV示例文件和统计报告已保存到 {csv_dir}/ 目录")
        
        # 打印下一步指南
        print("\n" + "=" * 60)
        print("下一步指南:")
        print("=" * 60)
        print("1. 查看CSV文件了解数据划分情况:")
        print(f"   - 数据划分汇总: {csv_dir}/data_split_summary.txt")
        print(f"   - 数据集统计: {csv_dir}/datasets_statistics.csv")
        print(f"   - 每日样本数: {csv_dir}/daily_sample_counts.csv")
        print(f"   - 数据示例: {csv_dir}/*_sample.csv")
        
        print("\n2. 模型训练阶段:")
        print("   - 使用 './data/train_features.pkl' 和 './data/train_labels.pkl' 进行训练")
        print("   - 使用 './data/val_features.pkl' 和 './data/val_labels.pkl' 进行验证")
        
        print("\n3. 回测阶段:")
        print("   - 使用 './data/test_features.pkl', './data/test_labels.pkl'")
        print("     和 './data/test_prices.pkl' 进行样本外回测")
        
        print("\n重要提醒: 确保三个阶段使用对应的数据集，避免数据泄露!")
        print("=" * 60)
        
        # 打印关键统计数据
        print("\n关键统计数据:")
        print("-" * 40)
        print(f"训练集占比: {train_features.shape[0]/features_df.shape[0]:.1%}")
        print(f"验证集占比: {val_features.shape[0]/features_df.shape[0]:.1%}")
        print(f"测试集占比: {test_features.shape[0]/features_df.shape[0]:.1%}")
        print(f"训练集平均每日样本数: {train_features.shape[0]/train_end_idx:.0f}")
        print(f"验证集平均每日样本数: {val_features.shape[0]/(val_end_idx - train_end_idx):.0f}")
        print(f"测试集平均每日样本数: {test_features.shape[0]/(total_days - val_end_idx):.0f}")
        
        return features_df, labels_series
        
    except Exception as e:
        print(f"\n数据准备过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 主程序入口
# __name__变量用于判断当前脚本是否作为主程序运行,
# 其他脚本import该脚本时，如果__name__的值不是"__main__"，则不会执行main()函数
if __name__ == "__main__":
    # 执行数据准备
    features_data, labels_data = main()
    
    # 如果数据准备成功，可以在这里进行简单的验证
    if features_data is not None:
        print("\n" + "=" * 60)
        print("数据验证:")
        print("1. 确保特征和标签的索引对齐:")
        if isinstance(labels_data, pd.DataFrame):
            common_idx = features_data.index.intersection(labels_data.index)
        else:
            common_idx = features_data.index.intersection(labels_data.index if hasattr(labels_data, 'index') else pd.Index([]))
        
        print(f"   对齐的样本数量: {len(common_idx)}")
        
        print("2. 检查特征维度:")
        print(f"   特征数量: {features_data.shape[1]}")
        
        print("3. 检查数据时间范围:")
        if hasattr(features_data.index, 'levels'):
            dates = features_data.index.get_level_values(0)
            print(f"   开始日期: {dates.min()}")
            print(f"   结束日期: {dates.max()}")
            print(f"   总交易日数: {len(dates.unique())}")
        
        print("=" * 60)