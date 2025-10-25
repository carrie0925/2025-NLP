import pandas as pd
import numpy as np

# ===== 可調參數 =====
SRC_CSV    = "arithmetic_train.csv"          # 原始 CSV
OUT_CSV    = "arithmetic_train_noisy.csv"    # 輸出（已加噪的完整資料）
LOG_CSV    = "noise_change_log.csv"          # 僅含被改動列（已是錯誤答案）
LABEL_COL  = "tgt"                           # 你的 label 欄位名稱（只改這一欄）
NOISE_RATE = 0.20                            # 20%
SEED       = 42                              # 固定隨機種子，確保可重現
# 噪聲規則：這裡用「+1」保證與原值不同、且不影響欄位型態
# 若要更亂，可改成 offsets = np.random.randint(1, 10, size=n_noise) * np.random.choice([1, -1], size=n_noise)
# 再用 df.loc[idx, LABEL_COL] += offsets
# ====================

rng = np.random.default_rng(SEED)

# 讀取
df = pd.read_csv(SRC_CSV)
orig_cols = list(df.columns)  # 保留原始欄位順序

if LABEL_COL not in df.columns:
    raise ValueError(f"找不到 label 欄位 '{LABEL_COL}'。現有欄位：{list(df.columns)}")

# 只轉換 label 欄為數值（不改其他欄位）
try:
    labels_numeric = pd.to_numeric(df[LABEL_COL])
except Exception as e:
    raise ValueError(f"無法將 '{LABEL_COL}' 轉為數值，請確認資料內容。錯誤：{e}")

n_total  = len(df)
n_noise  = int(n_total * NOISE_RATE)  # 恰好 20%（向下取整）
noise_idx = np.sort(rng.choice(n_total, size=n_noise, replace=False))

# 產生「錯誤」的 label（這裡採用 +1，保證不同且不改欄位型態）
df_noisy = df.copy()
df_noisy.loc[noise_idx, LABEL_COL] = labels_numeric.iloc[noise_idx].to_numpy() + 1

# === 產出檔案 ===
# 1) 全檔（不新增任何欄位，維持原欄位順序）
df_noisy = df_noisy.reindex(columns=orig_cols)
df_noisy.to_csv(OUT_CSV, index=False)

# 2) 僅含「被改動後」的整列資料（方便對照）
changed_rows = df_noisy.iloc[noise_idx].copy()
changed_rows.to_csv(LOG_CSV, index=False)

print(f"總筆數：{n_total}")
print(f"已異動（加噪）筆數：{n_noise}（{NOISE_RATE*100:.0f}%）")
print(f"輸出：{OUT_CSV}")
print(f"異動清單：{LOG_CSV}")
