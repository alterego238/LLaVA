# LLaVA 训练调试指南

## 调试配置说明

我已经为你创建了 `.vscode/launch.json` 文件，包含4个不同的调试配置：

### 1. Debug LLaVA Training (Single GPU)
- **用途**: 单GPU调试训练过程
- **特点**: 
  - 使用单GPU (CUDA_VISIBLE_DEVICES=0)
  - 较小的batch size (4) 适合调试
  - 保存频率较高 (每100步)
  - 关闭wandb报告

### 2. Debug LLaVA Training (Multi-GPU DeepSpeed)
- **用途**: 多GPU + DeepSpeed调试
- **特点**:
  - 使用双GPU (CUDA_VISIBLE_DEVICES=0,1)
  - 启用DeepSpeed ZeRO-3
  - 较大的batch size (8)
  - 适合测试分布式训练

### 3. Debug LLaVA Training (Small Batch)
- **用途**: 最小配置调试
- **特点**:
  - 最小batch size (1)
  - 关闭gradient checkpointing
  - 减少dataloader workers
  - 适合内存受限环境

### 4. Debug Data Loading Only
- **用途**: 仅测试数据加载
- **特点**:
  - 只运行1步训练
  - 关闭模型保存
  - 关闭lazy preprocessing
  - 适合调试数据管道

## 使用方法

1. **设置断点**: 在代码中设置断点
2. **选择配置**: 在VSCode调试面板选择相应的配置
3. **开始调试**: 按F5或点击开始调试按钮

## 常用调试位置

### 数据加载调试
- `llava/train/train.py:710` - `LazySupervisedDataset.__getitem__`
- `llava/train/train.py:760` - `DataCollatorForSupervisedDataset.__call__`

### 模型初始化调试
- `llava/train/train.py:806` - `train()` 函数开始
- `llava/train/train.py:874` - 模型梯度设置

### 训练循环调试
- `llava/train/llava_trainer.py` - 训练器实现

## 环境变量设置

在调试前，建议设置以下环境变量：

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
```

## 调试技巧

1. **内存监控**: 使用 `nvidia-smi` 监控GPU内存使用
2. **日志级别**: 设置 `--logging_steps 1` 查看详细日志
3. **数据验证**: 使用 "Debug Data Loading Only" 配置验证数据格式
4. **梯度检查**: 在训练循环中检查梯度值

## 常见问题

### 内存不足
- 减少 `per_device_train_batch_size`
- 关闭 `gradient_checkpointing`
- 减少 `model_max_length`

### 数据加载错误
- 检查数据路径是否正确
- 验证图像文件是否存在
- 检查JSON数据格式

### CUDA错误
- 设置 `CUDA_LAUNCH_BLOCKING=1` 获取详细错误信息
- 检查GPU驱动和CUDA版本兼容性

## 修改配置

如需修改调试参数，编辑 `.vscode/launch.json` 中的 `args` 数组：

```json
"args": [
    "--per_device_train_batch_size", "2",  // 修改batch size
    "--model_max_length", "1024",          // 修改序列长度
    "--save_steps", "50"                   // 修改保存频率
]
``` 