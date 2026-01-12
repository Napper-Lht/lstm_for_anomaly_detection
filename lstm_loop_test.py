import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import joblib  # 用于保存 scaler
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 全局配置
# ==========================================
TRAIN_FILE = "normal_train_fixed.csv"  # 你的干净训练数据
MODEL_PATH = "lstm_model.keras"  # 模型保存路径
SCALER_PATH = "scaler.pkl"  # 归一化参数保存路径
Config_PATH = "threshold.txt"  # 阈值保存路径

GROUP_SIZE = 400
TIME_STEPS = 2
NUM_TEST_LOOPS = 100  # 循环检测次数
SPIKE_ANGLE = 10.0  # 故障注入角度 (度)
NOISE_LEVEL = 0.0001  # 给测试集额外添加的高斯噪声强度 (模拟传感器热噪)

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 1. 工具函数 (数据处理 & 物理方程)
# ==========================================
def quat_normalize(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q


def quat_multiply(q, r):
    """ 四元数乘法 q ⊗ r """
    return np.array([
        q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
        q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
        q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
        q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
    ])


def random_spike_quaternion(angle_rad):
    """ 生成一个随机轴的旋转四元数 """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
    ])


def create_residual_dataset(X, group_size=400):
    Xs, ys_residual, ys_actual = [], [], []
    num_groups = int(np.ceil(len(X) / group_size))
    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(X))
        group = X[start:end]
        if len(group) <= 1: continue

        inputs = group[:-1, :]
        targets_next = group[1:, :12]
        targets_curr = group[:-1, :12]
        diff = targets_next - targets_curr

        Xs.append(inputs.reshape(-1, 1, inputs.shape[1]))
        ys_residual.append(diff)
        ys_actual.append(targets_next)

    if len(Xs) > 0:
        return np.concatenate(Xs), np.concatenate(ys_residual), np.concatenate(ys_actual)
    else:
        return np.array([]), np.array([]), np.array([])


def suppress_echo_anomalies(pred_labels, loss_values, attenuation_factor=1.2):
    """ 回声抑制后处理 """
    refined_labels = pred_labels.copy()
    for i in range(1, len(pred_labels)):
        if refined_labels[i - 1] == 1 and refined_labels[i] == 1:
            if loss_values[i] < loss_values[i - 1] * attenuation_factor:
                refined_labels[i] = 0
    return refined_labels


# ==========================================
# 2. 核心模块: 故障注入器 (Memory Mode)
# ==========================================
def inject_anomaly_in_memory(df_original, angle_deg, add_noise=True):
    """
    在内存中直接对 DataFrame 副本注入故障，不读写磁盘，速度快。
    """
    df = df_original.copy()
    N = len(df)

    # 1. 添加背景高斯噪声 (让测试更严苛)
    if add_noise:
        noise = np.random.normal(0, NOISE_LEVEL, df.shape)
        # 仅对数值列添加噪声 (假设全是数值)
        df = df + noise

    # 2. 随机选择注入点 (避开开头结尾)
    idx = np.random.randint(10, N - 10)

    # 3. 计算四元数突变
    angle_rad = angle_deg * np.pi / 180.0
    spike_q = random_spike_quaternion(angle_rad)

    # 4. 对三组星敏同时注入
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    for g in groups:
        # 提取当前四元数
        col_names = df.columns[g]
        q = df.iloc[idx, g].values.astype(float)
        q = quat_normalize(q)

        # 叠加突变
        q_new = quat_multiply(spike_q, q)
        q_new = quat_normalize(q_new)

        # 赋值回 DataFrame
        df.iloc[idx, g] = q_new

    # 5. 生成标签
    labels = np.zeros(N)
    # 注意：因为我们的模型是预测 (t -> t+1)，所以在 diff 模式下，
    # idx 处的突变会导致 idx-1 (预测 idx) 和 idx (预测 idx+1) 两个位置的 Loss 变大
    # 这里的标签我们需要标记 idx
    labels[idx] = 1

    return df, labels, idx


# ==========================================
# 3. 阶段一: 训练与封存 (Train & Save)
# ==========================================
def train_and_save_pipeline():
    print("\n" + "=" * 50)
    print(" >>> PHASE 1: Training Model & Determining Threshold")
    print("=" * 50)

    # 1. 加载训练数据
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return False

    train_df = pd.read_csv(TRAIN_FILE)

    # 2. 归一化 & 保存 Scaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✔ Scaler saved to {SCALER_PATH}")

    # 3. 构建数据集
    X_train, y_train_diff, _ = create_residual_dataset(train_scaled, GROUP_SIZE)

    # 4. 训练模型
    model = keras.Sequential([
        keras.Input(shape=(1, 16)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(12)
    ])
    model.compile(optimizer='adam', loss='mse')

    print("Training LSTM...")
    model.fit(X_train, y_train_diff, epochs=15, batch_size=32, verbose=1, validation_split=0.1)
    model.save(MODEL_PATH)
    print(f"✔ Model saved to {MODEL_PATH}")

    # 5. 确定阈值 (使用训练集的最大误差 x 安全系数，或者 IQR)
    # 为了演示 BF 搜索的有效性，我们这里先生成一个“验证故障集”来定阈值
    print("Generating Validation Set for Threshold Calibration...")
    val_df, val_labels, _ = inject_anomaly_in_memory(train_df, SPIKE_ANGLE, add_noise=False)
    val_scaled = scaler.transform(val_df)
    X_val, _, y_val_actual = create_residual_dataset(val_scaled, GROUP_SIZE)

    # 预测验证集
    y_pred_diff = model.predict(X_val, verbose=0)
    y_val_pred_restored = X_val[:, 0, :12] + y_pred_diff
    val_loss = np.mean(np.abs(y_val_pred_restored - y_val_actual), axis=1)

    # 对齐标签
    aligned_labels = []
    num_groups = int(np.ceil(len(val_labels) / GROUP_SIZE))
    for i in range(num_groups):
        start = i * GROUP_SIZE
        end = min((i + 1) * GROUP_SIZE, len(val_labels))
        g = val_labels[start:end]
        if len(g) > 1: aligned_labels.extend(g[1:])
    y_val_labels = np.array(aligned_labels)

    # --- BF Search 寻找最佳阈值 ---
    best_f1 = 0
    best_thresh = 0
    search_space = np.linspace(np.min(val_loss), np.max(val_loss), 1000)

    for t in search_space:
        # 这里用简单逻辑，不加回声抑制，为了速度
        f1 = f1_score(y_val_labels, (val_loss > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # 安全修正：如果 BF 搜出来的阈值太低（贴近噪声），强制拉高
    avg_noise = np.mean(val_loss[y_val_labels == 0])
    if best_thresh < avg_noise * 5:
        print("⚠️ BF Threshold too low, adjusting to 5 * Noise Level.")
        best_thresh = avg_noise * 5

    print(f"✔ Calibrated Threshold (BF Search): {best_thresh:.6f} (F1 on Val: {best_f1:.4f})")

    with open(Config_PATH, "w") as f:
        f.write(str(best_thresh))

    return True


# ==========================================
# 4. 阶段二: 循环评估 (Loop Evaluation)
# ==========================================
def run_evaluation_loop():
    print("\n" + "=" * 50)
    print(f" >>> PHASE 2: Running {NUM_TEST_LOOPS} Random Injections")
    print("=" * 50)

    # 1. 加载资源
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(Config_PATH)):
        print("Error: Model/Scaler/Threshold not found. Run Phase 1 first.")
        return

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(Config_PATH, "r") as f:
        THRESHOLD = float(f.read().strip())

    print(f"Loaded Threshold: {THRESHOLD:.6f}")

    # 加载原始数据作为注入模板
    raw_df = pd.read_csv(TRAIN_FILE)

    results = {
        "tp_count": 0,  # 成功检测到的异常数
        "fp_count": 0,  # 误报的点数
        "fn_count": 0,  # 漏报的异常数
        "detected_indices": []
    }

    history_fp = []

    print(f"Starting loop... (Total: {NUM_TEST_LOOPS})")

    for i in range(NUM_TEST_LOOPS):
        # A. 注入故障 (包含随机位置 + 随机高斯噪声)
        # 注意：这里我们每次生成一个新的随机异常
        test_df, test_labels_raw, anomaly_idx = inject_anomaly_in_memory(raw_df, SPIKE_ANGLE, add_noise=True)

        # B. 预处理
        test_scaled = scaler.transform(test_df)
        X_test, _, y_test_actual = create_residual_dataset(test_scaled, GROUP_SIZE)

        # 对齐标签
        y_labels = []
        num_groups = int(np.ceil(len(test_labels_raw) / GROUP_SIZE))
        for g_idx in range(num_groups):
            start = g_idx * GROUP_SIZE
            end = min((g_idx + 1) * GROUP_SIZE, len(test_labels_raw))
            g = test_labels_raw[start:end]
            if len(g) > 1: y_labels.extend(g[1:])
        y_labels = np.array(y_labels)

        # C. 预测
        y_pred_diff = model.predict(X_test, verbose=0)
        y_pred_restored = X_test[:, 0, :12] + y_pred_diff
        mae_loss = np.mean(np.abs(y_pred_restored - y_test_actual), axis=1)

        # D. 判定 & 回声抑制
        pred_labels = (mae_loss > THRESHOLD).astype(int)
        pred_labels_refined = suppress_echo_anomalies(pred_labels, mae_loss)

        # E. 统计 TP/FP
        # 当前测试集中只有 1 个真正的异常点 (anomaly_idx 附近)
        # 我们检查那个点的标签是否为 1

        tn, fp, fn, tp = confusion_matrix(y_labels, pred_labels_refined, labels=[0, 1]).ravel()

        results["tp_count"] += tp
        results["fp_count"] += fp
        results["fn_count"] += fn
        history_fp.append(fp)

        if (i + 1) % 10 == 0:
            print(f"Step {i + 1}/{NUM_TEST_LOOPS}: TP={tp}, FP={fp} | Inj_Idx={anomaly_idx}")

    # ==========================================
    # 5. 最终报告
    # ==========================================
    print("\n" + "=" * 50)
    print("       FINAL EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Tests Run   : {NUM_TEST_LOOPS}")
    print(f"Total Anomalies   : {NUM_TEST_LOOPS} (1 per test)")
    print(f"Successful Detects: {results['tp_count']} (TP)")
    print(f"Missed Detects    : {results['fn_count']} (FN)")
    print(f"False Alarms (Pts): {results['fp_count']} (Total FP points across all tests)")
    print(f"Avg FP per Test   : {np.mean(history_fp):.2f}")

    recall = results['tp_count'] / (results['tp_count'] + results['fn_count'])
    print(f"\n>>> Overall Recall : {recall:.2%}")
    print(f">>> Threshold Used : {THRESHOLD:.6f}")

    if recall > 0.95 and np.mean(history_fp) < 1.0:
        print("\n✅ Model Passed Stress Test! (High Recall, Low False Alarm)")
    else:
        print("\n⚠️ Model needs improvement. Check Threshold or Noise Level.")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 第一次运行请开启 Phase 1，之后可以注释掉 Phase 1 直接跑 Phase 2
    # 1. 训练并保存 (如果是新数据或想重置模型，请设为 True)
    RUN_TRAINING = True

    if RUN_TRAINING:
        success = train_and_save_pipeline()
        if not success: exit()

    # 2. 运行多次循环测试
    run_evaluation_loop()