"""
量化Transformer项目 - 阶段2：模型训练 (已修复版本)
修复了ReduceLROnPlateau的verbose参数问题
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置设备 (优先使用M1 GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"PyTorch版本: {torch.__version__}")

# ==================== 1. 数据加载与序列化Dataset (修正版) ====================
class FinancialDataset(Dataset):
    """
    将处理好的特征和标签转换为模型可用的序列数据 (修正版)
    修复了因数据交错存储导致的序列采样错误。
    """

    def __init__(self, features_df, labels_series, lookback_days=30, sample_step=1):
        super().__init__()
        self.lookback_days = lookback_days
        self.sample_step = sample_step

        # 1. 对齐索引
        self.common_index = features_df.index.intersection(labels_series.index)
        features_aligned = features_df.loc[self.common_index]
        labels_aligned = labels_series.loc[self.common_index]

        # 2. 核心修正：按股票代码分组，为每只股票单独存储数据
        self.stock_data_dict = {}  # 键：股票代码，值：该股票的特征数组 (形状: [该股票数据天数, 特征数])
        self.stock_label_dict = {}  # 键：股票代码，值：该股票的标签数组 (形状: [该股票数据天数])

        # 获取所有唯一的股票代码
        unique_stocks = features_aligned.index.get_level_values('股票代码').unique()

        for stock_code in unique_stocks:
            # 提取该只股票的所有特征和标签 (使用.xs方法，确保数据连续)
            stock_features = features_aligned.xs(stock_code, level='股票代码').values.astype(np.float32)
            stock_labels = labels_aligned.xs(stock_code, level='股票代码').values.astype(np.float32)

            # 存储到字典中
            self.stock_data_dict[stock_code] = stock_features
            self.stock_label_dict[stock_code] = stock_labels

        # 3. 基于新的数据结构创建样本列表
        self.samples = self._create_samples_v2()
        print(f"数据集创建完成: 总样本数 = {len(self.samples)}")
        print(f"特征维度: {features_df.shape[1]}, 序列长度: {lookback_days}")

    def _create_samples_v2(self):
        """为每只股票创建可用的样本序列 (修正版)"""
        samples = []

        for stock_code, features in self.stock_data_dict.items():
            labels = self.stock_label_dict[stock_code]
            num_days = len(features)  # 该只股票的总数据天数

            # 检查数据是否足够创建至少一个序列
            if num_days < self.lookback_days:
                continue

            # 在该股票连续的数据上创建滑动窗口
            for start in range(0, num_days - self.lookback_days, self.sample_step):
                end = start + self.lookback_days
                # 标签索引：假设标签对应序列结束日。请根据你的`create_labels`逻辑确认！
                # 如果你的标签是未来N日收益，这里应为 `end + N - 1`，但由于前面已经在标签创建时处理了未来N日，所以这里直接用end - 1
                label_idx = end - 1  # 使用序列最后一天的标签

                if label_idx < len(labels):
                    samples.append({
                        'stock_code': stock_code,
                        'feature_start': start,
                        'feature_end': end,  # 切片时使用 features[start:end]
                        'label_idx': label_idx
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        stock_code = sample_info['stock_code']

        # 从该股票的独立数组中截取特征序列
        # 形状确保为 [lookback_days, num_features]
        feature_seq = self.stock_data_dict[stock_code][sample_info['feature_start']:sample_info['feature_end']]

        # 获取对应的标签 (标量)
        label = self.stock_label_dict[stock_code][sample_info['label_idx']]

        return torch.FloatTensor(feature_seq), torch.FloatTensor([label])

    def get_stock_stats(self):
        """获取数据集统计信息"""
        return {
            'num_stocks': len(self.stock_data_dict),
            'total_samples': len(self.samples),
            'feature_dim': self.stock_data_dict[next(iter(self.stock_data_dict))].shape[1]
        }

# ==================== 2. 模型模块 ====================
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力层"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        #n_heads表示注意力头的数量，d_model表示模型的隐藏维度，d_k表示每个头的维度
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层，作用在张量的最后一个维度，输入是第一个维度，输出是最后一个维度。
        # 这里是Wq, Wk, Wv矩阵
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 预计算因果掩码（下三角矩阵）
        self.register_buffer("causal_mask", 
                            torch.triu(torch.full((1024, 1024), float('-inf')), diagonal=1))
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 线性投影并重塑为多头形式
        # q, k, v形状: [batch_size, n_heads, seq_len, d_k]
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数，最后两阶张量做矩阵乘法，最终形状: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 应用因果掩码，unsqueeze两次用于在最左侧增加2个维度以匹配attn_scores的形状，[batch_size, n_heads, seq_len, seq_len]
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        # 注意力权重，对最后一个维度softmax归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 上下文向量
        context = attn_weights @ v
        
        # 合并多头输出，contiguous用于确保内存连续，可以用.reshape()替代.contiguous().view()
        # 多头再接入全连接层输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(context)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # dim_feedforward表示前馈网络的隐藏层维度，通常是d_model的4倍
        #原因是金融数据序列通常具有以下特点：
        #1. 噪声大（市场波动）
        #2. 非线性关系复杂
        #3. 存在长期依赖
        #需要足够容量的网络来：
        #1. 过滤噪声
        #2. 捕捉复杂的非线性模式
        #3. 学习时间序列的长期依赖
        
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        # GELU的好处
        # 1. 平滑的过渡 → 梯度连续
        # 2. 保留负信息（缩放后） → 避免死亡神经元
        # 3. 概率门控 → 更符合注意力机制的思想
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力子层
        # 第一步：计算自注意力输出
        # 第二步：添加残差连接和层归一化
        # 第三步：通过前馈网络
        # 第四步：再次添加残差连接和层归一化
        # 最终输出经过两次子层处理的结果
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class FinancialTransformer(nn.Module):
    """用于金融序列预测的Transformer模型"""
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=3, 
                 dropout=0.1, use_calendar_feats=False):
        super().__init__()
        #d_model表示Transformer的隐藏维度，n_heads表示注意力头数，num_layers表示编码器层数
        #d_model=64是一个折中的选择，既能捕捉复杂模式，又不会过拟合，是可以调节的超参数
        
        # 经验法则：根据数据量选择模型大小
        #data_samples = 10000  # 假设有1万个训练样本

        #model_sizes = {
        #'非常小': (32, 1e5),    # 10万参数
        #'小': (64, 1e6),        # 100万参数
        #'中等': (128, 5e6),     # 500万参数
        #'大': (256, 2e7),       # 2000万参数
        #}

        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征投影层
        self.feature_proj = nn.Linear(input_dim, d_model)
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层，最后输出一个标量预测值
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # 初始化参数
        self._init_parameters()
        
        print(f"模型初始化: input_dim={input_dim}, d_model={d_model}, "
              f"layers={num_layers}, heads={n_heads}")
    
    def _init_parameters(self):
        """初始化模型参数"""
        """Xavier均匀初始化：
        从均匀分布 U(-a, a) 中采样，其中 a = gain * sqrt(6 / (fan_in + fan_out))"""

        #Xavier初始化的优势：
        #保持方差稳定：在前向和反向传播中保持激活值的方差
        #适合对称激活函数：如tanh、sigmoid，也适用于GELU
        #深层网络友好：防止梯度消失或爆炸

        for p in self.parameters():
            if p.dim() > 1: # 只初始化矩阵，不初始化向量（偏置）
                nn.init.xavier_uniform_(p) # Xavier均匀初始化
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 1. 特征投影
        x = self.feature_proj(x)  # [batch, seq_len, d_model]
        
        # 2. 添加位置编码，用前seq_len个位置编码，用广播机制加到x上
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # 3. Transformer编码器层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 4. 取最后一个时间步作为序列表示，它集成了过去所有时间步的信息
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 5. 输出预测，最右边的阶数被压缩到1，然后squeeze(-1)去掉，最终输出一个标量
        output = self.output_layer(x).squeeze(-1)  # [batch_size]
        
        return output

# ==================== 3. 损失函数与评估指标 ====================
def information_coefficient(pred, target):
    """计算信息系数 (皮尔逊相关系数)"""
    pred_norm = pred - pred.mean()
    target_norm = target - target.mean()
    
    #注意，下面的标准差是总体标准差（/n），不是样本标准差（/n-1）
    covariance = (pred_norm * target_norm).mean()
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    
    #浮点运算可能导致略微超出[-1,1]，.clamp(-1.0, 1.0)用于限制IC值在[-1, 1]范围内
    if pred_std > 0 and target_std > 0:
        ic = covariance / (pred_std * target_std)
        return ic.clamp(-1.0, 1.0)
    else:
        return torch.tensor(0.0).to(pred.device)

def negative_ic_loss(pred, target):
    """负IC损失：最小化负IC，等同于最大化IC"""
    ic = information_coefficient(pred, target)
    return -ic

def rank_information_coefficient(pred, target):
    """计算秩信息系数 (斯皮尔曼相关系数)"""
    # 转换为排序
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))
    
    # 计算排序的皮尔逊相关系数
    return information_coefficient(pred_rank.float(), target_rank.float())

# ==================== 4. 训练器类 ====================
class TransformerTrainer:
    """模型训练器 (已修复verbose参数问题)"""
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-4, weight_decay=5e-2):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        # AdamW 是对 Adam 的改进，加入了正确的权重衰减（weight decay）
        # 传统Adam将权重衰减与梯度更新混合，AdamW将其分离

        # 公式对比：
        # Adam: w_t+1 = w_t - η*(m_t/(√v_t + ε) + λ*w_t)  ← 权重衰减与梯度混合
        # AdamW: w_t+1 = w_t - η*(m_t/(√v_t + ε)) - η*λ*w_t  ← 分离的权重衰减

        # 优势：更稳定的训练，更好的泛化性能
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器 (已移除verbose参数)
        # 作用：当验证集指标不再提升时，自动降低学习率
        # 参数解释：
        # - optimizer: 要调度的优化器
        # - mode='max': 因为我们希望最大化IC（越大越好）
        # - factor=0.5: 每次降低学习率时，新学习率 = 原学习率 * 0.5
        # - patience=5: 如果连续5个epoch验证指标没有改善，则降低学习率

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 训练历史记录，字典格式
        self.history = {
            'train_loss': [], 'train_ic': [],
            'val_ic': [], 'best_val_ic': -float('inf'),
            'learning_rates': []
        }
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        #squeeze()函数用于去掉targets张量中所有维度为1的阶
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features, targets = features.to(self.device), targets.to(self.device).squeeze()
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # 计算损失
            loss = negative_ic_loss(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 伪代码：
            #def optimizer_step(parameters, gradients, lr):
                #for param, grad in zip(parameters, gradients):
                    # AdamW的更新规则
                    #m = beta1 * m_prev + (1-beta1) * grad  # 一阶矩估计
                    #v = beta2 * v_prev + (1-beta2) * grad² # 二阶矩估计
                    #m_hat = m / (1 - beta1^t)  # 偏差修正
                    #v_hat = v / (1 - beta2^t)
                    
            # 参数更新（包含权重衰减）
            #param = param - lr * (m_hat / (√v_hat + ε) + weight_decay * param)

            # 参数更新
            self.optimizer.step()
            
            # 记录
            # .item()：获取Python数值（浮点数）
            # .detach()：创建不参与梯度计算的新张量
            total_loss += loss.item()
            all_preds.append(predictions.detach())
            all_targets.append(targets.detach())
            
            # 进度显示
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算本epoch指标
        # 使用torch.cat拼接
        avg_loss = total_loss / len(self.train_loader)
        epoch_preds = torch.cat(all_preds)
        epoch_targets = torch.cat(all_targets)
        epoch_ic = information_coefficient(epoch_preds, epoch_targets).item()
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_ic'].append(epoch_ic)
        
        return avg_loss, epoch_ic
    
    def validate(self):
        """验证集评估"""
        self.model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features, targets = features.to(self.device), targets.to(self.device).squeeze()
                predictions = self.model(features)
                
                all_preds.append(predictions)
                all_targets.append(targets)
        
        # 计算验证集IC，cat用于把多个batch的结果拼接起来
        val_preds = torch.cat(all_preds)
        val_targets = torch.cat(all_targets)
        val_ic = information_coefficient(val_preds, val_targets).item()
        
        # 计算秩IC
        val_rank_ic = rank_information_coefficient(val_preds, val_targets).item()
        
        self.history['val_ic'].append(val_ic)
        
        return val_ic, val_rank_ic
    
    def train(self, num_epochs=30, save_path='best_model.pth'):
        """完整训练循环"""
        print(f"开始训练，共{num_epochs}个epoch...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            print(f"  当前学习率: {current_lr:.6f}")
            
            # 训练
            train_loss, train_ic = self.train_epoch(epoch)
            
            # 验证
            val_ic, val_rank_ic = self.validate()
            
            # 更新学习率 (这里会自动根据val_ic调整)
            self.scheduler.step(val_ic)
            
            # 保存最佳模型
            if val_ic > self.history['best_val_ic']:
                self.history['best_val_ic'] = val_ic # 最佳验证损失
                torch.save({
                    'epoch': epoch, # 最佳epoch编号
                    'model_state_dict': self.model.state_dict(), # 模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(), # 优化器状态
                    'val_ic': val_ic, 
                    'train_loss': train_loss,
                    'train_ic': train_ic,
                }, save_path)
                print(f"  ✅ 保存最佳模型，Val IC: {val_ic:.4f}")
            
            # 打印epoch结果
            print(f"  Train Loss: {train_loss:.4f}, Train IC: {train_ic:.4f}")
            print(f"  Val IC: {val_ic:.4f}, Val Rank IC: {val_rank_ic:.4f}")
            print(f"  Best Val IC: {self.history['best_val_ic']:.4f}")
            
            # 如果学习率降低了，打印提示
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"  📉 学习率从 {current_lr:.6f} 降低到 {new_lr:.6f}")
        
        print("\n训练完成！")
        return self.history

# ==================== 5. 主执行函数 ====================
def main():
    """主执行函数（使用独立验证集版本）"""
    print("=" * 60)
    print("量化Transformer - 模型训练阶段 (严格数据划分版)")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 1. 加载训练数据
    print("\n1. 加载训练数据...")
    try:
        with open('./data/train_features.pkl', 'rb') as f:
            train_features = pickle.load(f)
        with open('./data/train_labels.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        
        print(f"   训练特征数据形状: {train_features.shape}")
        print(f"   训练标签数据形状: {train_labels.shape}")
        
        # 显示训练数据前几行
        print("\n   训练数据预览:")
        print(train_features.head())
        
    except FileNotFoundError as e:
        print(f"错误: 未找到训练数据文件 - {e}")
        print("请先运行数据准备阶段 (data_preparation.py)")
        return None, None
    except Exception as e:
        print(f"加载训练数据时出错: {e}")
        return None, None
    
    # 2. 创建训练数据集
    print("\n2. 创建训练数据集...")
    try:
        train_dataset = FinancialDataset(
            features_df=train_features,
            labels_series=train_labels,
            lookback_days=30,
            sample_step=1  # 训练时使用密集采样
        )
        
        # 训练集统计
        train_stats = train_dataset.get_stock_stats()
        print(f"   训练集样本数: {train_stats['total_samples']}")
        print(f"   训练集股票数: {train_stats['num_stocks']}")
        print(f"   特征维度: {train_stats['feature_dim']}")
        
    except Exception as e:
        print(f"创建训练数据集时出错: {e}")
        return None, None
    
    # 3. 加载并创建验证数据集（关键修改）
    print("\n3. 加载并创建验证数据集...")
    try:
        with open('./data/val_features.pkl', 'rb') as f:
            val_features = pickle.load(f)
        with open('./data/val_labels.pkl', 'rb') as f:
            val_labels = pickle.load(f)
        
        print(f"   验证特征数据形状: {val_features.shape}")
        print(f"   验证标签数据形状: {val_labels.shape}")
        
        # 创建独立的验证数据集
        val_dataset = FinancialDataset(
            features_df=val_features,
            labels_series=val_labels,
            lookback_days=30,
            sample_step=1  # 验证时使用密集采样
        )
        
        val_stats = val_dataset.get_stock_stats()
        print(f"   验证集样本数: {val_stats['total_samples']}")
        print(f"   验证集股票数: {val_stats['num_stocks']}")
        
        # 检查训练集和验证集的时间范围是否重叠
        train_dates = set(train_features.index.get_level_values(0))
        val_dates = set(val_features.index.get_level_values(0))
        
        if train_dates.intersection(val_dates):
            print("⚠️  警告: 训练集和验证集有日期重叠!")
        else:
            print("✅ 训练集和验证集时间上完全隔离")
        
        # 打印日期范围
        train_date_min = min(train_dates)
        train_date_max = max(train_dates)
        val_date_min = min(val_dates)
        val_date_max = max(val_dates)
        
        print(f"   训练集时间范围: {train_date_min.date()} 到 {train_date_max.date()}")
        print(f"   验证集时间范围: {val_date_min.date()} 到 {val_date_max.date()}")
        
    except FileNotFoundError as e:
        print(f"错误: 未找到验证数据文件 - {e}")
        print("请确保数据准备阶段生成了验证集文件")
        return None, None
    except Exception as e:
        print(f"创建验证数据集时出错: {e}")
        return None, None
    
    # 4. 创建数据加载器
    print("\n4. 创建数据加载器...")
    try:
        # 根据M1内存调整batch_size
        batch_size = 32
        print(f"   Batch大小: {batch_size}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 训练时打乱顺序
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # 验证时不打乱
            num_workers=0
        )
        
        print(f"   训练集批次数量: {len(train_loader)}")
        print(f"   验证集批次数量: {len(val_loader)}")
        
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        return None, None
    
    # 5. 初始化模型
    print("\n5. 初始化模型...")
    try:
        # 使用训练集的特征维度
        input_dim = train_stats['feature_dim']
        model = FinancialTransformer(
            input_dim=input_dim,
            d_model=64,      # 隐藏层维度
            n_heads=4,       # 注意力头数
            num_layers=3,    # Transformer层数
            dropout=0.2      # Dropout率
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数量: {trainable_params:,}")
        print(f"   模型结构:")
        print(f"     - 输入维度: {input_dim}")
        print(f"     - 隐藏维度: 64")
        print(f"     - 注意力头数: 4")
        print(f"     - Transformer层数: 3")
        print(f"     - Dropout率: 0.2")
        
    except Exception as e:
        print(f"初始化模型时出错: {e}")
        return None, None
    
    # 6. 创建训练器并开始训练
    print("\n6. 开始训练...")
    try:
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=5e-4,
            weight_decay=1e-2
        )
        
        # 训练模型
        history = trainer.train(
            num_epochs=30,  # 训练轮数
            save_path='./best_transformer_model.pth'
        )
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 7. 训练结果分析
    print("\n7. 训练结果分析")
    print("=" * 50)
    print(f"   最佳验证集IC: {history['best_val_ic']:.4f}")
    print(f"   最终训练IC: {history['train_ic'][-1]:.4f}")
    print(f"   最终验证IC: {history['val_ic'][-1]:.4f}")
    print(f"   训练损失: {history['train_loss'][-1]:.4f}")
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 训练损失
        axes[0, 0].plot(history['train_loss'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # IC对比
        axes[0, 1].plot(history['train_ic'], label='Train IC', alpha=0.7)
        axes[0, 1].plot(history['val_ic'], label='Val IC', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Information Coefficient')
        axes[0, 1].set_title('Train vs Validation IC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最佳IC标记
        best_epoch = np.argmax(history['val_ic'])
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].text(best_epoch, history['val_ic'][best_epoch], 
                       f'Best: {history["val_ic"][best_epoch]:.4f}', 
                       color='r', fontsize=9)
        
        # 学习率变化
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 验证IC分布直方图
        axes[1, 1].hist(history['val_ic'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=np.mean(history['val_ic']), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(history["val_ic"]):.4f}')
        axes[1, 1].axvline(x=np.median(history['val_ic']), color='g', linestyle='--', 
                          label=f'Median: {np.median(history["val_ic"]):.4f}')
        axes[1, 1].set_xlabel('IC Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Validation IC Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Transformer量化模型训练', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('./training_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n   训练分析图已保存为 'training_analysis.png'")
        
    except ImportError:
        print("   Matplotlib未安装，跳过绘图")
    except Exception as e:
        print(f"   绘图时出错: {e}")
    
    print("\n" + "=" * 60)
    print("训练阶段完成！")
    print("下一步：使用训练好的模型在测试集上生成Alpha因子并进行回测")
    print("=" * 60)
    
    return model, history

#因为其他文件调用此代码的时候，__name__一般不是__main__，所以不会自动运行main函数
if __name__ == "__main__":
    # 运行主函数
    trained_model, training_history = main()