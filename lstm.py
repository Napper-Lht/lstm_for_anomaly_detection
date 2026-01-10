import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, \
    precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. 数据加载与预处理
# ==========================================
train_df = pd.read_csv('svd_corrected_train.csv')
test_df = pd.read_csv('test.csv')

test_labels = test_df['label'].values
test_data_for_pred = test_df.drop(columns=['label'])

scaler = StandardScaler()
scaler.fit(train_df)
# 注意：即使是残差学习，输入特征依然需要归一化
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_data_for_pred)

# [关键修正 1] 严格遵守物理约束：Time Steps = 1
# 更新只依赖上一拍，不需要看历史 20 步
GROUP_SIZE = 400
TIME_STEPS = 1


def create_residual_dataset(X, group_size=400):
    """
    构造残差学习数据:
    Input: [q_t, omega_t]
    Target: q_{t+1} - q_t (预测增量)
    """
    Xs, ys_residual, ys_actual = [], [], []
    num_groups = int(np.ceil(len(X) / group_size))

    print(f"处理数据: {len(X)} 行, {num_groups} 组 (TimeSteps=1, Residual Mode)")

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(X))
        group = X[start:end]

        if len(group) <= 1: continue

        # 输入: t 时刻 (包含 四元数 和 陀螺仪)
        # 输出: (t+1 时刻四元数) - (t 时刻四元数)
        # 注意：这里我们直接用归一化后的值做差分，这是机器学习的常用 trick
        # 只要能从 X_t 映射到 diff_t 即可
        inputs = group[:-1, :]
        targets_next = group[1:, :12]  # 下一刻四元数
        targets_curr = group[:-1, :12]  # 当前四元数

        diff = targets_next - targets_curr

        Xs.append(inputs.reshape(-1, 1, inputs.shape[1]))  # LSTM 需要 3D 输入
        ys_residual.append(diff)
        ys_actual.append(targets_next)  # 存下来用于评估验证

    return np.concatenate(Xs), np.concatenate(ys_residual), np.concatenate(ys_actual)


X_train, y_train_diff, _ = create_residual_dataset(train_scaled, GROUP_SIZE)
X_test, y_test_diff, y_test_actual = create_residual_dataset(test_scaled, GROUP_SIZE)


# 对齐标签 (因为每组少了一个点)
def align_labels_steps1(labels, group_size=400):
    aligned = []
    num_groups = int(np.ceil(len(labels) / group_size))
    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(labels))
        g = labels[start:end]
        if len(g) > 1:
            aligned.extend(g[1:])  # 预测的是 t+1，所以标签从第 2 个开始
    return np.array(aligned)


y_test_labels = align_labels_steps1(test_labels, GROUP_SIZE)

# ==========================================
# 2. 构建“物理感知”模型
# ==========================================
# 结构：简单的 MLP (或者单层 LSTM) 即可，因为是马尔可夫过程
model = keras.Sequential([
    keras.Input(shape=(1, 16)),
    keras.layers.LSTM(64, return_sequences=False),  # 也可以换成 Dense，LSTM 对序列有一点平滑作用
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(12)  # 输出 12 维的 DELTA (增量)
])

model.compile(optimizer='adam', loss='mse')

print(">>> Training Residual Model...")
history = model.fit(
    X_train, y_train_diff,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
    shuffle=True  # TimeSteps=1 时可以打乱，打破时间相关性，让模型专注学习物理方程
)

# ==========================================
# 3. 预测与还原
# ==========================================
print(">>> Predicting...")
# 1. 模型预测增量 delta
y_pred_diff = model.predict(X_test)

# 2. 还原预测值: Pred_Next = Curr + Predicted_Delta
# X_test[:, 0, :12] 是当前时刻的四元数输入
y_test_pred_restored = X_test[:, 0, :12] + y_pred_diff

# 3. 计算误差 (还原后的值 vs 真实下一刻的值)
test_mae_loss = np.mean(np.abs(y_test_pred_restored - y_test_actual), axis=1)

# ==========================================
# 4. [新增] 查缺补漏调试模块
# ==========================================
print("\n" + "=" * 40)
print("       DIAGNOSTIC REPORT (调试报告)")
print("=" * 40)

# A. 物理基准检查 (Naive Baseline Check)
# 假设：如果我不动，预测值就是当前值 (Delta=0)。如果模型的误差比这个还大，说明模型坏了。
naive_loss = np.mean(np.abs(X_test[:, 0, :12] - y_test_actual), axis=1)
avg_model_loss = np.mean(test_mae_loss)
avg_naive_loss = np.mean(naive_loss)

print(f"1. 模型效能检查:")
print(f"   - Naive Baseline Error (不做预测): {avg_naive_loss:.5f}")
print(f"   - Your Model Error     (智能预测): {avg_model_loss:.5f}")
if avg_model_loss < avg_naive_loss:
    print("   [PASS] 模型成功学到了物理规律 (误差小于基准)。")
else:
    print("   [FAIL] 警告！模型不如直接猜测“数值不变”。请检查数据或归一化。")

# B. 寻找最佳阈值
precisions, recalls, thresholds = precision_recall_curve(y_test_labels, test_mae_loss)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
BEST_THRESHOLD = thresholds[best_idx]

# C. 生成最终指标
test_pred_labels = (test_mae_loss > BEST_THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_labels, test_pred_labels).ravel()

print(f"\n2. 最终检测指标 (基于残差模型):")
print(f"   Threshold: {BEST_THRESHOLD:.4f}")
print(f"   F1 Score : {f1_scores[best_idx]:.4f}")
print(f"   Recall   : {recall_score(y_test_labels, test_pred_labels):.4f}")
print(f"   Precision: {precision_score(y_test_labels, test_pred_labels):.4f}")
print(f"   TP: {tp}  FN: {fn} (漏报)")
print(f"   FP: {fp}  TN: {tn}")

# ==========================================
# 5. 可视化 FP (误报) 案例
# ==========================================
# 看看模型为什么会误报？是不是因为角速度太大？
fp_indices = np.where((test_pred_labels == 1) & (y_test_labels == 0))[0]

if len(fp_indices) > 0:
    plt.figure(figsize=(12, 5))
    # 取误差最大的前 50 个 FP 点
    top_fps = fp_indices[np.argsort(test_mae_loss[fp_indices])[-50:]]

    # 计算这些点的角速度模长 (输入特征的第12-15列是陀螺仪)
    # X_test 形状 (N, 1, 16) -> 取 [:, 0, 12:]
    gyro_mag_fp = np.linalg.norm(X_test[top_fps, 0, 12:], axis=1)

    plt.scatter(range(len(top_fps)), gyro_mag_fp, color='orange', label='Gyro Magnitude of FP samples')
    plt.axhline(np.mean(np.linalg.norm(X_test[:, 0, 12:], axis=1)), color='blue', linestyle='--',
                label='Average Gyro Mag (All Data)')
    plt.title("Diagnostics: Do False Positives happen at high rotation speed?")
    plt.ylabel("Gyro Magnitude (Normalized)")
    plt.xlabel("Sample Index (Top 50 FPs)")
    plt.legend()
    plt.show()
    print("\n[调试提示] 请查看上方弹出的散点图：")
    print("如果橙色点普遍高于蓝色虚线，说明模型在【快速旋转】时容易误报。")
    print("应对措施：增加训练集中高动态数据的权重，或对高动态区域放宽阈值。")
else:
    print("恭喜！没有 False Positive，模型非常完美。")