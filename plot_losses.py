import os
import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalars, weight=0.95):  # 指数加权移动平均
    """
    Smoothing loss curves using exponential moving average
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 加载损失记录
loss_save_path = './logs/ciao/0111_1152_temp.json'
with open(loss_save_path, 'r') as fs:
    loss_history = json.load(fs)

# 创建保存图像的目录
plot_save_dir = './logs/ciao/'
os.makedirs(plot_save_dir, exist_ok=True)

# 设置全局字体样式
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# 所有要绘制的损失项
loss_keys = ['sum_loss', 'bpr_loss', 'reg_loss', 'uue_loss', 'iue_loss','cl_loss']
titles = {
    'sum_loss': 'Total Loss',
    'bpr_loss': 'BPR Loss',
    'reg_loss': 'Regularization Loss',
    'uue_loss': 'Social Graph Diffusion Loss',
    'iue_loss': 'User-Item Graph Diffusion Loss',
    'cl_loss': 'contrastive_loss'
}

# 单独绘图：每个 loss 单独一张图
for key in loss_keys:
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history[key], label='Raw', alpha=0.5)
    plt.plot(smooth(loss_history[key]), label='Smoothed', linewidth=2)
    plt.title(titles[key])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plot_save_path = os.path.join(plot_save_dir, f'{key}_curve.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 合并绘图：将所有 loss 放在一张图中进行比较
plt.figure(figsize=(12, 6))
for key in loss_keys:
    plt.plot(loss_history[key], label=titles[key], alpha=0.6)
plt.title("All Losses Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plot_save_path = os.path.join(plot_save_dir, "all_losses_curve.png")
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 所有图像已保存至: {plot_save_dir}")
