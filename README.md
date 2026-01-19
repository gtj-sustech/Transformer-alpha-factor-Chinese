# Transformer-alpha-factor-Chinese
基于Transformer的量化交易策略（30日特征输入，70/15/15时间划分），以信息系数(IC)为核心优化目标。模型采用d_model=64/n_heads=4/num_layers=3架构，预测未来5日收益率。回测采用月度调仓多头策略（因子前20%选股，0.1%手续费），年化收益18.64%（基准10.5%），IC均值0.0324（IR=2.145），最大回撤&lt;15%。项目提供完整数据处理、模型训练、回测分析闭环，生成可视化报告与绩效指标，为量化策略开发提供可复现标准化框架。
# 项目结构
'''
├── data_preparation.py       # 数据准备阶段（生成特征和标签）
├── model_pipeline_fixed.py   # 模型训练阶段（训练Transformer模型）
├── factor_backtest.py        # 因子生成与回测阶段
├── data/                     # 保存生成的数据
│   ├── train_features.pkl
│   ├── train_labels.pkl
│   ├── val_features.pkl
│   ├── val_labels.pkl
│   ├── test_features.pkl
│   ├── test_labels.pkl
│   └── transformer_quant_data_prices.pkl
└── backtest_results/         # 回测结果
    ├── backtest_report.txt
    └── backtest_charts.png
'''
    
# 项目特点
严格时间划分：训练集、验证集、测试集按时间严格划分，避免数据泄露
IC优化：以信息系数(IC)为优化目标，直接提升预测能力
多维度分析：包含因子IC分析、回测绩效指标、最大回撤等全面评估
M1 Mac优化：针对Apple Silicon设备进行了后端优化
完整回测流程：从特征生成到策略回测的完整闭环
# 快速开始
1. 安装依赖
   pip install akshare pandas numpy torch matplotlib scikit-learn
2. 数据准备
   python data_preparation.py
3. 模型训练
   python model_pipeline_fixed.py
4. 回测分析
   python factor_backtest.py
结果示例
   策略最终净值: 1,245,678.90 元
   基准最终净值: 1,050,000.00 元
   策略超额收益: 195,678.90 元
   策略相对收益: 18.64%

   因子IC均值: 0.0324
   因子信息比率(IR): 2.145
# 项目优势
1. 数据安全：严格时间划分，避免未来数据泄露
2. 预测能力：以IC为优化目标，直接提升因子预测能力
3. 回测严谨：包含交易成本、调仓频率等实际交易因素
4. 结果可视化：提供直观的净值曲线、回撤曲线、IC序列等图表
