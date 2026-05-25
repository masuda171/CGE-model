import numpy as np
from scipy.optimize import brentq

#パラメーターの設定
alpha = {'BRD': 0.3, 'MLK': 0.7}

beta_base = {
    ('CAP', 'BRD'): 1/3,
    ('LAB', 'BRD'): 2/3,
    ('CAP', 'MLK'): 4/7,
    ('LAB', 'MLK'): 3/7,
}

b = {
    'BRD': 3 / (2 ** (2/3)),
    'MLK': 7 / (4 ** (4/7) * 3 ** (3/7)),
}

FF = {'CAP': 25, 'LAB': 25}

#均衡計算関数の設定（ニュメレールはp_LAB_f=1.0とする）
def compute_equilibrium(p_CAP_f, beta, p_LAB_f=1.0):
    p_f = {'CAP': p_CAP_f, 'LAB': p_LAB_f}

#財価格の計算  
    p = {}
    for j in ['BRD', 'MLK']:
        p[j] = (1 / b[j]) * np.prod([
            (p_f[h] / beta[(h, j)]) ** beta[(h, j)]
            for h in ['CAP', 'LAB']
        ])

#家計所得と消費量
    I = sum(p_f[h] * FF[h] for h in ['CAP', 'LAB'])
    X = {i: alpha[i] * I / p[i] for i in ['BRD', 'MLK']}
    Z = X.copy()  # 財市場均衡：Z_j = X_i

#要素需要の計算
    F = {(h, j): beta[(h, j)] * p[j] * Z[j] / p_f[h]
         for h in ['CAP', 'LAB'] for j in ['BRD', 'MLK']}

#超過需要の計算
    EDL = F[('LAB', 'BRD')] + F[('LAB', 'MLK')] - FF['LAB']
    EDK = F[('CAP', 'BRD')] + F[('CAP', 'MLK')] - FF['CAP']

    return p, p_f, I, X, Z, F, EDL, EDK

#ベータの変更
beta_new = beta_base.copy()
beta_new[('CAP', 'BRD')] = 0.5
beta_new[('LAB', 'BRD')] = 0.5

p_tmp, pf_tmp, I_tmp, X_tmp, Z_tmp, F_tmp, EDL_tmp, EDK_tmp = \
    compute_equilibrium(1.0, beta_new)

print(f"EDL = {EDL_tmp:.4f}, EDK = {EDK_tmp:.4f}")
# → EDL = -2.5000, EDK = 2.5000

#solverで均衡を求める
def EDL_func(p_CAP_f):
    _, _, _, _, _, _, EDL, _ = compute_equilibrium(p_CAP_f, beta_new)
    return EDL

p_CAP_new = brentq(EDL_func, 0.01, 100.0)


# 新均衡の全変数を計算
p1, pf1, I1, X1, Z1, F1, EDL1, EDK1 = compute_equilibrium(p_CAP_new, beta_new)

# 結果の出力
print(f"\n--- 新均衡 ---")
print(f"p_CAP_f' = {p_CAP_new:.4f}")
print(f"p_BRD    = {p1['BRD']:.4f}")
print(f"p_MLK    = {p1['MLK']:.4f}")
print(f"I        = {I1:.4f}")
print(f"X_BRD    = {X1['BRD']:.4f}")
print(f"X_MLK    = {X1['MLK']:.4f}")
print(f"EDL      = {EDL1:.8f}")
print(f"EDK      = {EDK1:.8f}")

# ベースライン均衡を計算
p0, pf0, I0, X0, Z0, F0, _, _ = compute_equilibrium(1.0, beta_base)

# (7) 消費量の変化
print(f"\n--- 問2(7) 消費量の変化 ---")
print(f"ΔX_BRD = {X1['BRD']:.4f} - {X0['BRD']:.4f} = {X1['BRD']-X0['BRD']:+.4f}")
print(f"ΔX_MLK = {X1['MLK']:.4f} - {X0['MLK']:.4f} = {X1['MLK']-X0['MLK']:+.4f}")

# (8) 等価変分 EV
U0 = np.prod([X0[i] ** alpha[i] for i in ['BRD', 'MLK']])
U1 = np.prod([X1[i] ** alpha[i] for i in ['BRD', 'MLK']])

def expenditure(prices, U):
    return U * np.prod([(prices[i] / alpha[i]) ** alpha[i] for i in ['BRD', 'MLK']])

EV = expenditure(p0, U1) - I0

print(f"\n--- 問2(8) 等価変分 EV ---")
print(f"U_base = {U0:.4f}")
print(f"U_new  = {U1:.4f}")
print(f"EV     = {EV:+.4f}")
if EV < 0:
    print("→ EV < 0：技術変化により家計の厚生は悪化した")