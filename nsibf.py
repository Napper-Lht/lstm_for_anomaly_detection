import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# ==========================================
# 0. 核心配置
# ==========================================
WINDOW_SIZE = 2
LATENT_DIM = 8
BATCH_SIZE = 64
EPOCHS = 20  # 快速验证设为20即可
GROUP_SIZE = 400

TARGET_COLS = list(range(12))
AUX_COLS = list(range(12, 16))

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 1. 数据诊断工具 (关键!)
# ==========================================
def diagnose_data(df, name="Test Data"):
    print(f"\n>>> Diagnosing {name}...")
    if 'label' not in df.columns:
        print("   No label column found. Skipping diagnosis.")
        return

    anom_rows = df[df['label'] == 1]
    if len(anom_rows) == 0:
        print("   ⚠️ WARNING: Label column exists but contains NO anomalies (all 0)!")
        return

    print(f"   Found {len(anom_rows)} labeled anomaly points.")

    # 检查第一个异常点的真实突变情况
    idx = anom_rows.index[0]
    if idx > 0:
        # 计算该点和上一点的欧氏距离 (仅看四元数部分 0-11)
        prev_row = df.iloc[idx - 1, TARGET_COLS].values
        curr_row = df.iloc[idx, TARGET_COLS].values

        # 简单的欧式距离
        diff = np.linalg.norm(curr_row - prev_row)

        # 计算全局平均跳变幅度作为基准
        # 随机抽样 1000 个点
        sample_diffs = []
        for _ in range(1000):
            r = np.random.randint(1, len(df))
            if df.iloc[r]['label'] == 0:
                d = np.linalg.norm(df.iloc[r, TARGET_COLS].values - df.iloc[r - 1, TARGET_COLS].values)
                sample_diffs.append(d)
        avg_diff = np.mean(sample_diffs)

        print(f"   [Anomaly Check] Row {idx}")
        print(f"   -> Signal Jump (Delta): {diff:.6f}")
        print(f"   -> Normal Noise Level : {avg_diff:.6f}")

        if diff < avg_diff * 2:
            print("\n   🔴🔴 CRITICAL WARNING 🔴🔴")
            print("   The labeled anomaly is indistinguishable from normal noise!")
            print("   Please check your 'filechange.py' or data generation logic.")
            print("   If the data values aren't modified, the model CANNOT detect it.")
            print("   🔴🔴 CRITICAL WARNING 🔴🔴\n")
        else:
            print(f"   ✅ Data looks valid. Anomaly is {diff / avg_diff:.1f}x larger than noise.")


# ==========================================
# 2. 连续化与构建
# ==========================================
def make_quaternion_continuous(data):
    data = data.copy()
    for start_col in [0, 4, 8]:
        qs = data[:, start_col: start_col + 4]
        for i in range(1, len(qs)):
            if np.dot(qs[i], qs[i - 1]) < 0:
                qs[i] = -qs[i]
        data[:, start_col: start_col + 4] = qs
    return data


def create_dataset(X, window_size=2, group_size=400):
    x_prev_list, win_list, u_curr_list, x_delta_list = [], [], [], []
    indices = []  # 记录对应的原始索引

    all_idx = np.arange(X.shape[1])
    is_u = np.isin(all_idx, AUX_COLS)

    num_groups = int(np.ceil(len(X) / group_size))
    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(X))
        group = X[start:end]
        if len(group) <= window_size: continue

        for j in range(window_size, len(group)):
            x_prev = group[j - 1][~is_u]
            window = group[j - window_size: j]
            u_curr = group[j][is_u]

            x_curr = group[j][~is_u]
            x_delta = x_curr - x_prev

            x_prev_list.append(x_prev)
            win_list.append(window)
            u_curr_list.append(u_curr)
            x_delta_list.append(x_delta)

            # 记录这是第几行数据 (绝对索引)
            indices.append(start + j)

    return (np.array(x_prev_list, dtype=np.float32),
            np.array(win_list, dtype=np.float32),
            np.array(u_curr_list, dtype=np.float32),
            np.array(x_delta_list, dtype=np.float32),
            np.array(indices))


# ==========================================
# 3. NSIBF 模型
# ==========================================
class NSIBF_Optimized(keras.Model):
    def __init__(self, x_dim, u_dim, z_dim):
        super(NSIBF_Optimized, self).__init__()
        self.g_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(z_dim, activation='tanh')
        ], name='g_net')
        self.lstm_layer = layers.LSTM(32, activation='tanh')
        self.f_dense = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(z_dim, activation='tanh')
        ], name='f_net_dense')
        self.h_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(x_dim, activation='linear')
        ], name='h_net')

    def call(self, inputs):
        x_prev, win_seq, u_curr = inputs
        z_prev = self.g_net(x_prev)
        lstm_feat = self.lstm_layer(win_seq)
        f_in = layers.Concatenate()([z_prev, lstm_feat, u_curr])
        z_pred = self.f_dense(f_in)
        x_delta_next = self.h_net(z_pred)
        return x_delta_next, z_prev, z_pred


# ==========================================
# 4. 主流程
# ==========================================
print(">>> Loading Data...")
try:
    train_df = pd.read_csv('normal_train_fixed.csv')
    test_df = pd.read_csv('test.csv')
except:
    train_df = pd.DataFrame(np.random.randn(2000, 16), columns=[f'f{i}' for i in range(16)])
    test_df = pd.DataFrame(np.random.randn(500, 16), columns=[f'f{i}' for i in range(16)])
    test_df['label'] = 0
    test_df.iloc[250, -1] = 1  # Dummy label

# --- 步骤 1: 诊断数据 ---
diagnose_data(test_df, "Test CSV")

print(">>> Preprocessing...")
# 连续化
train_cont = make_quaternion_continuous(train_df.values)
if 'label' in test_df.columns:
    test_values = test_df.drop(columns=['label']).values
    test_labels_raw = test_df['label'].values
else:
    test_values = test_df.values
    test_labels_raw = np.zeros(len(test_df))
test_cont = make_quaternion_continuous(test_values)

# 归一化
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_cont).astype(np.float32)
test_scaled = scaler.transform(test_cont).astype(np.float32)

# 构建数据集
x_p_tr, win_tr, u_tr, delta_tr, _ = create_dataset(train_scaled, WINDOW_SIZE, GROUP_SIZE)
x_p_te, win_te, u_te, delta_te, idx_te = create_dataset(test_scaled, WINDOW_SIZE, GROUP_SIZE)

# 对齐标签
y_test_aligned = test_labels_raw[idx_te]

# 训练
nsibf = NSIBF_Optimized(x_dim=12, u_dim=4, z_dim=LATENT_DIM)
optimizer = keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(x_prev, win, u_curr, x_delta_target):
    with tf.GradientTape() as tape:
        x_delta_pred, _, z_pred = nsibf([x_prev, win, u_curr])
        x_next_recon = x_prev + x_delta_target
        z_next_true = nsibf.g_net(x_next_recon)
        l_pred = tf.reduce_mean(tf.square(x_delta_target - x_delta_pred))
        l_consist = tf.reduce_mean(tf.square(z_next_true - z_pred))
        total_loss = l_pred + 0.1 * l_consist
    grads = tape.gradient(total_loss, nsibf.trainable_weights)
    optimizer.apply_gradients(zip(grads, nsibf.trainable_weights))
    return l_pred


print(">>> Training...")
dataset = tf.data.Dataset.from_tensor_slices((x_p_tr, win_tr, u_tr, delta_tr)).shuffle(1000).batch(BATCH_SIZE)
for epoch in range(EPOCHS):
    loss_avg = 0
    for bx, bw, bu, by in dataset:
        loss_avg += train_step(bx, bw, bu, by)
    if (epoch + 1) % 5 == 0: print(f"Epoch {epoch + 1} Loss: {loss_avg / len(dataset):.6f}")

# 检测
print(">>> Detecting...")
delta_pred_te, _, _ = nsibf([x_p_te, win_te, u_te])
# 使用简单的 MSE 评分 (最鲁棒)
scores = np.mean(np.square(delta_te - delta_pred_te.numpy()), axis=1)

# 计算阈值
delta_pred_tr, _, _ = nsibf([x_p_tr, win_tr, u_tr])
scores_tr = np.mean(np.square(delta_tr - delta_pred_tr.numpy()), axis=1)
threshold = np.percentile(scores_tr, 99.9)
print(f"Threshold: {threshold:.6f}")


# ==========================================
# 5. 点调整评估 (Point Adjustment) - 解决对齐误差
# ==========================================
def evaluate_with_adjustment(y_true, y_scores, threshold):
    pred_raw = (y_scores > threshold).astype(int)

    # Point Adjustment: 如果一个异常区间内有任意一点被检测到，则整个区间算检测到
    # 对于单点异常，这意味着如果 label=1 的点被检测到，就 OK。
    # 这里我们做一个简化版：如果 label=1 的前后 1 个点内有报警，就算 TP

    pred_adjusted = pred_raw.copy()
    actual_anoms = np.where(y_true == 1)[0]

    for idx in actual_anoms:
        # 检查该点及其前后是否有一点报警
        window_check = pred_raw[max(0, idx - 1): min(len(pred_raw), idx + 2)]
        if np.sum(window_check) > 0:
            pred_adjusted[idx] = 1  # 修正为检测成功

    return pred_adjusted


pred_final = evaluate_with_adjustment(y_test_aligned, scores, threshold)
tn, fp, fn, tp = confusion_matrix(y_test_aligned, pred_final, labels=[0, 1]).ravel()

print("\n" + "=" * 40)
print("       FINAL DIAGNOSTIC REPORT")
print("=" * 40)
print(f"TP: {tp} (Caught) | FN: {fn} (Missed)")
print(f"FP: {fp} (False)  | TN: {tn}")
print("-" * 40)

if tp > 0:
    print(f"✅ Success! Anomaly caught.")
    # 找到那个被抓到的异常点
    caught_idx = np.where((y_test_aligned == 1) & (pred_final == 1))[0]
    print(f"   Score at anomaly: {scores[caught_idx[0]]:.6f} (Threshold: {threshold:.6f})")
else:
    print("❌ Failed.")
    # 分析漏报原因
    if np.sum(y_test_aligned) > 0:
        real_anom_idx = np.where(y_test_aligned == 1)[0][0]
        real_score = scores[real_anom_idx]
        print(f"   Score at anomaly index ({real_anom_idx}): {real_score:.8f}")
        print(f"   Threshold is {threshold:.8f}. The score is too low.")
        print("   -> Check the 'Data Diagnosis' section at the top of this log.")

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(scores, label='Error Score', color='blue', alpha=0.6)
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
anom_indices = np.where(y_test_aligned == 1)[0]
if len(anom_indices) > 0:
    plt.scatter(anom_indices, scores[anom_indices], color='red', s=100, label='True Anomaly', zorder=10)
plt.yscale('log')
plt.title("Anomaly Detection Score")
plt.legend()
plt.show()