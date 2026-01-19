"""
量化Transformer项目 - 阶段3：因子生成与回测
M1 Mac优化版
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime, timedelta
import os
from scipy import stats
import matplotlib

# M1 Mac后端优化
try:
    # 尝试使用MacOSX原生后端
    matplotlib.use('MacOSX')
    print("使用MacOSX后端")
except:
    try:
        # 备选：Agg后端（非交互式，最稳定）
        matplotlib.use('Agg')
        print("使用Agg后端")
    except:
        pass

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体和警告
warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # macOS中文显示
rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 1. 模型重新加载 ====================
def load_trained_model(model_path, input_dim, d_model=64, n_heads=4, num_layers=3):
    #model_path:模型权重文件的路径（字符串）
    """加载训练好的模型"""
    from model_pipeline_fixed import FinancialTransformer  # 导入之前定义的模型类
    
    # 初始化模型结构
    model = FinancialTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=0.0  # 推理时不需要dropout
    )
    
    # 加载训练好的权重
    #checkpoint['model_state_dict']：从检查点文件中提取模型状态字典、包含模型所有层的权重和偏置参数
    #model.load_state_dict()：将加载的权重设置到模型实例中、使模型具有训练好的参数

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) # 将模型移动到指定设备
    model.eval()  # 设置为评估模式
    
    print(f"模型加载成功: {model_path}")
    print(f"训练时的最佳验证IC: {checkpoint.get('val_ic', 'N/A'):.4f}")
    
    return model

# ==================== 2. 因子生成器 ====================
class AlphaFactorGenerator:
    """将Transformer预测转换为Alpha因子"""
    
    def __init__(self, model, lookback_days=30):
        self.model = model
        self.lookback_days = lookback_days
        self.device = device
            
    def prepare_sequences(self, features_df, target_date, stock_codes=None):
        """为指定日期准备所有股票的输入序列 (修正版)"""
        if target_date not in features_df.index.get_level_values(0):
            return None, None
        
        # 获取该日期前 lookback_days 个交易日（包含target_date）
        # 确保取到的是连续的交易日
        all_dates = sorted(features_df.index.get_level_values(0).unique())
        
        # 找到 target_date 在 all_dates 中的位置
        try:
            date_idx = all_dates.index(target_date)
        except ValueError:
            return None, None
        
        if date_idx < self.lookback_days - 1:  # 需要有足够的过去数据
            return None, None
        
        # 确定时间窗口 [start_date, target_date] 共 lookback_days 天
        window_dates = all_dates[date_idx - self.lookback_days + 1: date_idx + 1]
        
        # 如果指定了股票列表，则只处理这些股票
        if stock_codes is None:
            stock_codes = features_df.index.get_level_values(1).unique()
        
        sequences = []
        valid_stocks = []
        
        for stock_code in stock_codes:
            try:
                # 一次性获取该股票在完整时间窗口内的所有数据
                stock_df = features_df.xs(stock_code, level='股票代码')
                
                # 确保数据按日期排序
                stock_df = stock_df.sort_index()
                
                # 获取窗口数据 - 使用 .loc 和日期范围
                window_data = stock_df.loc[window_dates[0]:window_dates[-1]]
                
                # 检查是否正好有 lookback_days 天的数据
                if len(window_data) == self.lookback_days:
                    # 取特征值 (排除可能存在的索引列)
                    features = window_data.values
                    sequences.append(features)
                    valid_stocks.append(stock_code)
                    
            except Exception as e:
                continue  # 该股票数据不足或有问题，跳过
        
        if len(sequences) == 0:
            return None, None
        
        # 转换形状为 [股票数量, lookback_days, 特征维度] 的张量
        sequences_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        return sequences_tensor, valid_stocks
    
  
    def generate_factors_for_date(self, features_df, date):
        """为单个日期生成Alpha因子"""
        
        #prepare_sequences是前面定义的数据准备函数
        sequences, stock_codes = self.prepare_sequences(features_df, date)
        
        if sequences is None:
            return pd.Series(dtype=float), []
        
        # 批量预测
        with torch.no_grad():
            predictions = self.model(sequences)
        
        # 结果转回CPU，转换为numpy数组
        predictions_np = predictions.cpu().numpy()
        
        # 创建因子Series
        factor_series = pd.Series(predictions_np, index=stock_codes, name=date)
        
        return factor_series, stock_codes
    
    def generate_factor_series(self, features_df, start_date=None, end_date=None):
        """生成时间序列的Alpha因子"""
        
        # 获取所有日期
        # get_level_values(0)用于获取第一层索引的值放入列表中
        # unique()用于去除重复的值
        all_dates = sorted(features_df.index.get_level_values(0).unique())
        
        # 设置日期范围
        if start_date is not None:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date is not None:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        
        # 需要足够的历史数据
        all_dates = all_dates[self.lookback_days:]
        
        print(f"开始生成因子，共{len(all_dates)}个交易日...")
        
        # 初始化存储字典和进度列表
        factor_data = {}
        date_progress = []
        
        # 每20个交易日打印一次进度
        ## factor_data 是一个字典
        #factor_data = {
        #    '2023-01-01': pd.Series([1.2, -0.5, 0.8], index=['AAPL', 'MSFT', 'GOOGL']),
        #    '2023-01-02': pd.Series([0.9, -0.3, 1.1], index=['AAPL', 'MSFT', 'GOOGL']),
        #    ...
        #}
        for i, date in enumerate(all_dates):
            if (i + 1) % 20 == 0 or i == 0 or i == len(all_dates) - 1:
                print(f"  进度: {i+1}/{len(all_dates)} - {date.date()}")
            
            factor_series, _ = self.generate_factors_for_date(features_df, date)
            
            if len(factor_series) > 10:  # 至少要有10只股票
                # 横截面标准化
                factor_mean = factor_series.mean()
                factor_std = factor_series.std()
                
                if factor_std > 0:
                    # Z-score标准化
                    standardized = (factor_series - factor_mean) / factor_std
                    # Winsorize处理极端值
                    standardized = standardized.clip(-3, 3)
                    factor_data[date] = standardized
                    date_progress.append(date)
        
        print(f"因子生成完成，有效交易日: {len(factor_data)}")
        
        # 转换为DataFrame
        ## 1. 先创建 DataFrame
        #temp_df = pd.DataFrame(factor_data)
        # 输出：
        #            2023-01-01  2023-01-02
        # AAPL             1.2         0.9
        # MSFT            -0.5        -0.3
        # GOOGL            0.8         1.1
        # ↑ 此时行索引是股票，列索引是日期

        # 2. 转置后
        #factor_df = temp_df.T
        # 输出：
        #           AAPL  MSFT  GOOGL
        # 2023-01-01  1.2  -0.5    0.8
        # 2023-01-02  0.9  -0.3    1.1
        # ↑ 转置后：行索引是日期，列索引是股票
        if factor_data:
            factor_df = pd.DataFrame(factor_data).T  # 日期为索引，股票为列
            return factor_df
        else:
            return pd.DataFrame()
        
# ==================== 3. 回测引擎 ====================
class BacktestEngine:
    """策略回测引擎"""
    
    def __init__(self, factor_df, price_data, transaction_cost=0.001, initial_capital=1000000):
        """
        参数:
            factor_df: Alpha因子DataFrame (日期×股票)
            price_data: 价格DataFrame (日期×股票)
            transaction_cost: 单边交易成本
            initial_capital: 初始资金
        """
        self.factor_df = factor_df
        self.price_data = price_data
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # 对齐日期，确保因子和价格数据在相同日期都有数据
        self.common_dates = factor_df.index.intersection(price_data.index)
        self.factor_df = factor_df.loc[self.common_dates]
        self.price_data = price_data.loc[self.common_dates]
        
        print(f"回测日期范围: {self.common_dates[0].date()} 到 {self.common_dates[-1].date()}")
        print(f"回测交易日数: {len(self.common_dates)}")
        print(f"股票数量: {self.factor_df.shape[1]}")
    
    def calculate_returns(self):
        """计算股票收益率"""
        # 使用日收益率（可根据需要调整频率）.pct_change()计算每日百分比变化
        returns = self.price_data.pct_change()
        returns = returns.loc[self.common_dates]  # 确保日期对齐
        
        # 填充NaN（第一天没有收益）
        returns = returns.fillna(0)
        
        return returns
    
    def long_only_portfolio(self, top_pct=0.2, rebalance_freq='M'):
        """
        纯多头组合策略（做多前top_pct%的股票）
        top_pct: 做多的股票比例（0.2表示前20%）
        """
        
        returns = self.calculate_returns()
        portfolio_returns = [] # 每日组合收益率
        positions_history = [] # 每日持仓记录
        capital_history = [] # 每日资金记录
        
        current_capital = self.initial_capital
        current_positions = {}  # 股票:持仓比例
        
        # 确定调仓日期（保持不变）
        if rebalance_freq == 'D':
            rebalance_dates = self.common_dates
        elif rebalance_freq == 'W':
            rebalance_dates = []
            for date in self.common_dates:
                if date.weekday() == 4:  # 周五
                    rebalance_dates.append(date)
        elif rebalance_freq == 'M':
            rebalance_dates = []
            current_month = None
            for date in self.common_dates:
                if date.month != current_month:
                    rebalance_dates.append(date)
                    current_month = date.month
        
        print(f"调仓频率: {rebalance_freq}, 调仓次数: {len(rebalance_dates)}")
        print(f"做多比例: 前{top_pct*100}%的股票")
        
        for i, date in enumerate(self.common_dates):
            if i == 0:
                # 第一天，没有持仓
                daily_return = 0
                portfolio_returns.append(daily_return)
                positions_history.append({})
                capital_history.append(current_capital)
                continue
            
            # 计算当日组合收益
            if current_positions:
                daily_return = 0
                for stock, weight in current_positions.items():
                    if stock in returns.columns and pd.notna(returns.loc[date, stock]):
                        daily_return += weight * returns.loc[date, stock]
            else:
                daily_return = 0
            
            portfolio_returns.append(daily_return)
            capital_history.append(current_capital)
            
            # 更新资金
            current_capital *= (1 + daily_return)
            
            # 检查是否需要调仓
            if date in rebalance_dates:
                if date in self.factor_df.index:
                    # 获取当日的因子值
                    day_factors = self.factor_df.loc[date]
                    
                    # 移除NaN值
                    valid_factors = day_factors.dropna()
                    
                    if len(valid_factors) >= 20:  # 至少要有20只有效股票
                        # 按因子值排序，ascending=False表示从大到小排序
                        sorted_stocks = valid_factors.sort_values(ascending=False)
                        
                        # 选择做多的股票（前top_pct%）
                        n_stocks = len(sorted_stocks)
                        long_count = max(1, int(n_stocks * top_pct))
                        
                        # 只选择做多股票
                        long_stocks = sorted_stocks.head(long_count).index.tolist()
                        
                        # 计算权重（等权重，100%资金做多）
                        long_weight = 1.0 / len(long_stocks)
                        
                        # 构建新持仓（只做多）
                        new_positions = {}
                        for stock in long_stocks:
                            new_positions[stock] = long_weight
                        
                        # 计算换手率和交易成本
                        turnover = self.calculate_turnover(current_positions, new_positions)
                        transaction_cost_amount = turnover * current_capital * self.transaction_cost
                        
                        # 扣除交易成本
                        current_capital -= transaction_cost_amount
                        
                        current_positions = new_positions
                
                positions_history.append(current_positions.copy())
            else:
                positions_history.append(current_positions.copy())
        
        # 计算组合净值
        portfolio_nav = [self.initial_capital]
        for ret in portfolio_returns[1:]:
            portfolio_nav.append(portfolio_nav[-1] * (1 + ret))
        
        results = {
            'dates': self.common_dates,
            'returns': pd.Series(portfolio_returns, index=self.common_dates),
            'nav': pd.Series(portfolio_nav, index=self.common_dates),
            'capital': pd.Series(capital_history, index=self.common_dates),
            'positions': positions_history,
            'rebalance_dates': rebalance_dates
        }
        
        return results
    
    def calculate_turnover(self, old_positions, new_positions):
        """计算换手率"""
        if not old_positions:
            return 1.0  # 初始建仓，100%换手
        
        all_stocks = set(list(old_positions.keys()) + list(new_positions.keys()))
        turnover = 0.0
        
        for stock in all_stocks:
            old_weight = old_positions.get(stock, 0) # get()方法用于获取字典中指定键的值
            new_weight = new_positions.get(stock, 0)
            turnover += abs(new_weight - old_weight) # 计算权重变化
        
        return turnover / 2  # 单边换手率
    
    def benchmark_portfolio(self, benchmark='equal_weight'):
        """基准策略"""
        returns = self.calculate_returns()
        
        if benchmark == 'equal_weight':
            # 等权重组合，axis=1表示按行（每日）计算均值
            benchmark_returns = returns.mean(axis=1)
        elif benchmark == 'hs300':
            # 这里可以添加沪深300基准
            # 简化：使用所有股票等权重作为基准
            benchmark_returns = returns.mean(axis=1)
        else:
            benchmark_returns = returns.mean(axis=1)
        
        # 计算基准净值
        benchmark_nav = [self.initial_capital] # 初始资金
        for ret in benchmark_returns.iloc[1:]:
            benchmark_nav.append(benchmark_nav[-1] * (1 + ret)) # 计算净值
        
        return pd.Series(benchmark_returns, index=self.common_dates), pd.Series(benchmark_nav, index=self.common_dates)

# ==================== 4. 绩效分析 ====================
class PerformanceAnalyzer:
    """绩效分析器"""
    
    @staticmethod
    def calculate_metrics(returns_series, nav_series, risk_free_rate=0.03):
        """计算绩效指标"""
        
        # 年化收益率
        total_days = len(returns_series)
        years = total_days / 252  # 假设252个交易日/年
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # 年化波动率
        annualized_vol = returns_series.std() * np.sqrt(252)
        
        # 夏普比率
        if annualized_vol > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown, drawdown_duration = PerformanceAnalyzer.calculate_max_drawdown(nav_series)
        
        # Calmar比率
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = annualized_return
        
        # 胜率
        positive_days = (returns_series > 0).sum()
        win_rate = positive_days / len(returns_series) if len(returns_series) > 0 else 0
        
        # 盈亏比
        avg_win = returns_series[returns_series > 0].mean() if (returns_series > 0).any() else 0
        avg_loss = returns_series[returns_series < 0].mean() if (returns_series < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        metrics = {
            '总收益率': total_return,
            '年化收益率': annualized_return,
            '年化波动率': annualized_vol,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            'Calmar比率': calmar_ratio,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '总交易日': total_days
        }
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(nav_series):
        """计算最大回撤"""
        nav_array = nav_series.values
        peak = np.maximum.accumulate(nav_array)
        drawdown = (peak - nav_array) / peak
        
        max_dd = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        # 寻找回撤开始点（前一个峰值）
        peak_idx = np.argmax(nav_array[:max_dd_idx+1])
        
        # 计算回撤持续时间
        if max_dd > 0:
            drawdown_duration = max_dd_idx - peak_idx
        else:
            drawdown_duration = 0
        
        return max_dd, drawdown_duration
    
    @staticmethod
    def calculate_factor_metrics(factor_df, future_returns):
        """计算因子相关指标"""
        
        ic_series = []
        ic_dates = []
        
        # 对齐日期
        common_dates = factor_df.index.intersection(future_returns.index)
        
        for date in common_dates:
            # 获取当天的因子值和下期收益
            if date in factor_df.index and date in future_returns.index:
                # 获取共同股票
                factor_series = factor_df.loc[date]
                returns_series = future_returns.loc[date]
                
                common_stocks = factor_series.index.intersection(returns_series.index)
                
                if len(common_stocks) >= 10:  # 至少10只股票
                    factor_values = factor_series[common_stocks]
                    return_values = returns_series[common_stocks]
                    
                    # 计算IC（秩相关系数）
                    ic = factor_values.corr(return_values, method='spearman')
                    
                    if not pd.isna(ic):
                        ic_series.append(ic)
                        ic_dates.append(date)
        
        if len(ic_series) == 0:
            return {}
        
        ic_series = pd.Series(ic_series, index=ic_dates)
        
        # 计算IC统计
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # IC显著性检验
        if len(ic_series) > 1:
            t_stat, p_value = stats.ttest_1samp(ic_series, 0)
        else:
            t_stat, p_value = 0, 1
        
        factor_metrics = {
            'IC均值': ic_mean,
            'IC标准差': ic_std,
            '信息比率(IR)': ic_ir,
            'IC T统计量': t_stat,
            'IC P值': p_value,
            'IC>0比例': (ic_series > 0).mean(),
            '有效IC天数': len(ic_series)
        }
        
        return factor_metrics, ic_series
    
    @staticmethod
    def generate_report(strategy_results, benchmark_results, factor_metrics_tuple, output_path='./backtest_report'):
        """生成完整的回测报告 (修正版)"""
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 1. 绩效指标对比
        strategy_metrics = PerformanceAnalyzer.calculate_metrics(
            strategy_results['returns'], 
            strategy_results['nav']
        )
        
        benchmark_metrics = PerformanceAnalyzer.calculate_metrics(
            benchmark_results['returns'], 
            benchmark_results['nav']
        )
        
        # 2. 正确处理因子指标 (接受元组)
        factor_metrics_dict = {}
        ic_series = None
        
        if factor_metrics_tuple and len(factor_metrics_tuple) == 2:
            factor_metrics_dict, ic_series = factor_metrics_tuple
        
        # 3. 生成图表
        PerformanceAnalyzer.plot_results(
            strategy_results, 
            benchmark_results, 
            factor_metrics_dict, 
            ic_series,  # 明确传递 ic_series
            output_path
        )
        
        # 4. 生成文本报告 (移除对backtest_engine的引用)
        report_text = PerformanceAnalyzer.create_text_report(
            strategy_metrics, 
            benchmark_metrics, 
            factor_metrics_dict,
            initial_capital=strategy_results.get('initial_capital', 1000000)  # 从结果中获取
        )
        
        # 保存报告
        report_file = os.path.join(output_path, 'backtest_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"回测报告已保存到: {report_file}")
        
        return report_text
    
    @staticmethod
    def plot_results(strategy_results, benchmark_results, factor_metrics_dict, ic_series, output_path):
        """绘制回测结果图表 (修正版)"""
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建2x2的子图
            fig = plt.figure(figsize=(16, 12))
            
            # 1. 净值曲线对比
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(strategy_results['nav'].index, strategy_results['nav'].values, 
                    label='Transformer策略', linewidth=2, color='#2E86AB')
            ax1.plot(benchmark_results['nav'].index, benchmark_results['nav'].values, 
                    label='等权重基准', linewidth=1.5, color='#A23B72', alpha=0.7)
            ax1.set_title('净值曲线对比', fontsize=14, fontweight='bold')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('净值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 回撤曲线
            ax2 = plt.subplot(2, 2, 2)
            # 计算策略回撤
            nav_array = strategy_results['nav'].values
            peak = np.maximum.accumulate(nav_array)
            drawdown = (peak - nav_array) / peak
            
            ax2.fill_between(strategy_results['nav'].index, 0, drawdown * 100, 
                        color='#F18F01', alpha=0.5)
            ax2.plot(strategy_results['nav'].index, drawdown * 100, 
                    color='#C73E1D', linewidth=1)
            ax2.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('回撤 (%)')
            ax2.grid(True, alpha=0.3)
            
            # 3. IC序列 - 使用传递的ic_series参数
            if ic_series is not None and len(ic_series) > 0:
                ax3 = plt.subplot(2, 2, 3)
                ax3.bar(ic_series.index, ic_series.values * 100, 
                    color='#3BBA9C', alpha=0.6, width=1)
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                ic_mean = ic_series.mean()
                ax3.axhline(y=ic_mean * 100, color='red', 
                        linestyle='--', linewidth=1, label=f"均值: {ic_mean*100:.2f}%")
                ax3.set_title('IC序列 (日度)', fontsize=14, fontweight='bold')
                ax3.set_xlabel('日期')
                ax3.set_ylabel('IC (%)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. 月度收益热力图
            ax4 = plt.subplot(2, 2, 4)
            # 计算月度收益
            monthly_returns = strategy_results['returns'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # 转换为数据透视表格式
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            })
            
            pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            # 处理可能出现的空数据
            if not pivot_table.empty:
                im = ax4.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', 
                            vmin=-10, vmax=10)
                ax4.set_title('月度收益热力图 (%)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('月份')
                ax4.set_ylabel('年份')
                ax4.set_xticks(range(12))
                ax4.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月', 
                                    '7月', '8月', '9月', '10月', '11月', '12月'])
                
                # 添加颜色条
                plt.colorbar(im, ax=ax4, label='收益率 (%)')
            else:
                ax4.text(0.5, 0.5, '月度数据不足', ha='center', va='center', fontsize=12)
                ax4.set_title('月度收益热力图', fontsize=14, fontweight='bold')
            
            plt.suptitle('Transformer量化策略回测分析', fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(output_path, 'backtest_charts.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"回测图表已保存到: {chart_path}")
            
        except Exception as e:
            print(f"绘制图表时出错: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def create_text_report(strategy_metrics, benchmark_metrics, factor_metrics, initial_capital=1000000):
        """创建文本报告 (修正版)"""
        
        report = "=" * 80 + "\n"
        report += "TRANSFORMER量化策略回测报告\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 80 + "\n\n"
        
        # 1. 策略概览
        report += "1. 策略概览\n"
        report += "-" * 40 + "\n"
        report += f"回测期间: {strategy_metrics['总交易日']} 个交易日\n"
        report += f"初始资金: {initial_capital:,.2f} 元\n\n"  # 使用传入的参数
        
        # 2. 绩效指标对比
        report += "2. 绩效指标对比\n"
        report += "-" * 40 + "\n"
        report += "指标               策略               基准\n"
        report += "-" * 40 + "\n"
        
        metrics_to_show = ['总收益率', '年化收益率', '年化波动率', '夏普比率', 
                        '最大回撤', 'Calmar比率', '胜率', '盈亏比']
        
        for metric in metrics_to_show:
            strat_val = strategy_metrics[metric]
            bench_val = benchmark_metrics[metric]
            
            if metric in ['总收益率', '年化收益率', '年化波动率', '最大回撤']:
                strat_fmt = f"{strat_val:.2%}"
                bench_fmt = f"{bench_val:.2%}"
            elif metric in ['夏普比率', 'Calmar比率', '盈亏比']:
                strat_fmt = f"{strat_val:.3f}"
                bench_fmt = f"{bench_val:.3f}"
            elif metric == '胜率':
                strat_fmt = f"{strat_val:.2%}"
                bench_fmt = f"{bench_val:.2%}"
            else:
                strat_fmt = str(strat_val)
                bench_fmt = str(bench_val)
            
            report += f"{metric:10} {strat_fmt:>20} {bench_fmt:>20}\n"
        
        report += "\n"
        
        # 3. 因子评价
        if factor_metrics and len(factor_metrics) > 0:
            report += "3. 因子评价指标\n"
            report += "-" * 40 + "\n"
            
            for key, value in factor_metrics.items():
                if key == 'IC均值':
                    report += f"{key:15} {value:.4f} ({value*100:.2f}%)\n"
                elif key == '信息比率(IR)':
                    report += f"{key:15} {value:.3f}\n"
                elif key == 'IC P值':
                    report += f"{key:15} {value:.4f} {'(显著)' if value < 0.05 else '(不显著)'}\n"
                elif key in ['IC>0比例']:
                    report += f"{key:15} {value:.2%}\n"
                else:
                    report += f"{key:15} {value:.4f}\n"
            
            report += "\n"
        
        # 4. 结论
        report += "4. 结论与建议\n"
        report += "-" * 40 + "\n"
        
        # 自动生成结论
        if '夏普比率' in strategy_metrics and strategy_metrics['夏普比率'] > benchmark_metrics['夏普比率']:
            report += "✅ 策略夏普比率优于基准，风险调整后收益更好。\n"
        else:
            report += "⚠️  策略夏普比率低于基准，需进一步优化。\n"
        
        if '最大回撤' in strategy_metrics and strategy_metrics['最大回撤'] < benchmark_metrics['最大回撤']:
            report += "✅ 策略最大回撤小于基准，风险控制较好。\n"
        else:
            report += "⚠️  策略最大回撤较大，需加强风险管理。\n"
        
        if factor_metrics and 'IC均值' in factor_metrics:
            ic_mean = factor_metrics['IC均值']
            if ic_mean > 0.03:
                report += f"✅ 因子IC均值({ic_mean:.4f})较高，预测能力强。\n"
            elif ic_mean > 0.01:
                report += f"📊 因子IC均值({ic_mean:.4f})中等，有一定预测能力。\n"
            else:
                report += f"⚠️  因子IC均值({ic_mean:.4f})较低，预测能力有限。\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "报告结束\n"
        report += "=" * 80
        
        return report
    
# ==================== 5. 主执行函数 ====================
def main():
    """主执行函数：完整的因子生成与回测流程"""
    
    print("=" * 80)
    print("量化Transformer项目 - 阶段3: 因子生成与回测")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 1. 加载数据

        # features_df: 包含特征数据的DataFrame
        # 多级索引：第一层是日期，第二层是股票代码
        # 列是各种技术指标特征（如移动平均线、RSI等）
        # 形状示例：(总数据点数, 特征维度)
        # labels_series: 包含标签数据的Series，通常是未来收益率（如次日收益率），用于训练监督学习模型
        print("\n1. 加载数据...")
        # 修改数据加载部分
        with open('./data/test_features.pkl', 'rb') as f:
            features_df = pickle.load(f)
        with open('./data/test_labels.pkl', 'rb') as f:
            labels_series = pickle.load(f)

        # 加载测试集价格数据
        with open('./data/test_prices.pkl', 'rb') as f:
            price_data = pickle.load(f)
        
        print(f"   特征数据形状: {features_df.shape}")
        print(f"   标签数据形状: {labels_series.shape}")
        
        # 2. 加载训练好的模型
        print("\n2. 加载训练好的模型...")
        model_path = './best_transformer_model.pth'
        
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件 {model_path}")
            print("请先运行模型训练阶段 (model_pipeline_fixed.py)")
            return
        
        # input_dim：获取特征维度
        # features_df.shape[1] 返回列数（特征数量）
        # 例如：如果有20个技术指标，input_dim=20
        input_dim = features_df.shape[1]
        model = load_trained_model(model_path, input_dim)
        
        # 3. 生成Alpha因子
        print("\n3. 生成Alpha因子...")
        factor_generator = AlphaFactorGenerator(model, lookback_days=30)
        
        # 使用全部数据生成因子（不再拆分测试集）
        all_dates = sorted(features_df.index.get_level_values(0).unique())
        
        # 使用全部日期范围
        start_date = all_dates[0]  # 第一个日期
        end_date = all_dates[-1]   # 最后一个日期
        
        print(f"   生成因子日期范围: {start_date.date()} 到 {end_date.date()}")
        print(f"   总数据量: {len(all_dates)} 个交易日")
        
        factor_df = factor_generator.generate_factor_series(
            features_df, 
            start_date=start_date,  # 改为开始日期
            end_date=end_date       # 改为结束日期
        )
        
        if factor_df.empty:
            print("错误: 无法生成因子数据")
            return
        
        print(f"   生成的因子数据形状: {factor_df.shape}")
        
        # 4. 准备价格数据用于回测
        print("\n4. 准备回测数据...")
        try:
            # 尝试加载专门保存的价格数据
            prices_file = './data/transformer_quant_data_prices.pkl'
            if os.path.exists(prices_file):
                with open(prices_file, 'rb') as f:
                    price_data = pickle.load(f)
                print(f"   加载价格数据形状: {price_data.shape}")
                print(f"   价格数据日期范围: {price_data.index.min()} 到 {price_data.index.max()}")
            else:
                print("错误: 未找到价格数据文件，请先运行数据准备阶段并确保保存了价格数据。")
                return False
        except Exception as e:
            print(f"加载价格数据时出错: {e}")
            return False

        # 对齐价格数据和因子数据
        common_dates = factor_df.index.intersection(price_data.index)
        factor_df = factor_df.loc[common_dates]
        price_data = price_data.loc[common_dates]

        print(f"   回测数据: {len(common_dates)}个交易日, {factor_df.shape[1]}只股票")
        
        # 5. 运行回测
        print("\n5. 运行策略回测...")
        backtest_engine = BacktestEngine(
            factor_df=factor_df,
            price_data=price_data,
            transaction_cost=0.001,  # 千分之一交易成本
            initial_capital=1000000  # 100万初始资金
        )
        
        # 运行策略
        strategy_results = backtest_engine.long_only_portfolio(
            top_pct=0.2,  # 做多前20%的股票
            rebalance_freq='M'  # 每月调仓
        )
        
        # 基准策略
        benchmark_returns, benchmark_nav = backtest_engine.benchmark_portfolio('equal_weight')
        benchmark_results = {
            'returns': benchmark_returns,
            'nav': benchmark_nav
        }
        
        # 6. 因子评价
        print("\n6. 计算因子评价指标...")
        # 准备未来收益率数据
        future_returns = pd.pivot_table(
            labels_series.reset_index(),
            values='future_return',
            index='date',
            columns='股票代码'
        )
        
        # 计算因子指标
        factor_metrics = PerformanceAnalyzer.calculate_factor_metrics(factor_df, future_returns)
        
        # 7. 生成完整报告
        print("\n7. 生成回测报告...")
        report = PerformanceAnalyzer.generate_report(
            strategy_results, 
            benchmark_results, 
            factor_metrics,
            output_path='./backtest_results'
        )
        
        # 8. 打印关键结果
        print("\n" + "=" * 80)
        print("回测完成！关键结果:")
        print("=" * 80)
        
        strategy_nav = strategy_results['nav']
        benchmark_nav = benchmark_results['nav']
        
        final_strategy_value = strategy_nav.iloc[-1]
        final_benchmark_value = benchmark_nav.iloc[-1]
        
        print(f"策略最终净值: {final_strategy_value:,.2f} 元")
        print(f"基准最终净值: {final_benchmark_value:,.2f} 元")
        print(f"策略超额收益: {final_strategy_value - final_benchmark_value:,.2f} 元")
        print(f"策略相对收益: {(final_strategy_value/final_benchmark_value - 1)*100:.2f}%")
        
        if isinstance(factor_metrics, tuple) and len(factor_metrics) > 0:
            print(f"\n因子IC均值: {factor_metrics[0].get('IC均值', 'N/A'):.4f}")
            print(f"因子信息比率(IR): {factor_metrics[0].get('信息比率(IR)', 'N/A'):.3f}")
        
        print("\n详细报告请查看: ./backtest_results/backtest_report.txt")
        print("图表文件: ./backtest_results/backtest_charts.png")
        print("=" * 80)
        
        return True

    # traceback.print_exc() 打印完整的错误堆栈信息    
    except Exception as e:
        print(f"\n❌ 运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行完整回测
    success = main()
    

