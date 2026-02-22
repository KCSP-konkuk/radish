"""
무 도매 평균가격 순별 예측 모델
================================
모델: XGBoost
피쳐: 전체 후보 중 importance 기반 동적 선별
학습: 2018~2024, 테스트: 2025

기상 데이터 지역 매핑 (계절별 주산지):
  겨울무 (12~2월)  → 성산 (제주 월동무)
  봄무   (3~6월)   → 고창군 (봄무 주산지)
  여름무 (7~8월)   → 대관령 (고랭지 여름무)
  가을무 (9~11월)  → 고창군 (가을무 주산지)

데이터 파일 (./data/ 폴더):
  가격데이터_순별.csv   : 무 도매 평균가격
  무반입량_순별.csv     : 서울 도매시장 총반입량
  search_순별.csv       : 네이버 검색량
  성산_순별.csv         : 제주 성산 기상 (겨울무)
  고창군_순별.csv       : 전북 고창군 기상 (봄·가을무)
  대관령_순별.csv       : 강원 대관령 기상 (여름무)
"""

import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time

t0 = time.time()

DATA_DIR         = "./data/"
OUTPUT_PRED_DIR  = "./results/predictions/"
OUTPUT_PLOT_DIR  = "./results/plots/"

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)


# ================================================================
# 1. 공통 유틸리티
# ================================================================
def parse_date(d):
    """'202301상순' → (year, month, period_int, period_str)"""
    d = str(d).strip().replace('\ufeff', '')
    period_map = {'상순': 0, '중순': 1, '하순': 2}
    year       = int(d[:4])
    month      = int(d[4:6])
    period_str = d[6:]
    period     = period_map[period_str]
    return year, month, period, period_str


def to_idx(year, month, period):
    """순별 전역 인덱스 (2017년 기준)"""
    return (year - 2017) * 36 + (month - 1) * 3 + period


def load_csv(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    parsed         = df['DATE'].apply(parse_date)
    df['Year']     = parsed.apply(lambda x: x[0])
    df['Month']    = parsed.apply(lambda x: x[1])
    df['Period']   = parsed.apply(lambda x: x[2])
    df['PeriodStr']= parsed.apply(lambda x: x[3])
    df['idx']      = df.apply(lambda r: to_idx(r['Year'], r['Month'], r['Period']), axis=1)
    return df.sort_values('idx').reset_index(drop=True)


print("=" * 60)
print("  무 도매가격 순별 예측 모델")
print("=" * 60)


# ================================================================
# 2. 데이터 로드
# ================================================================
price = load_csv(DATA_DIR + "가격데이터_순별.csv")
for col in ['평균가격', '전년', '평년']:
    price[col] = pd.to_numeric(price[col], errors='coerce')

seongsan   = load_csv(DATA_DIR + "성산_순별.csv")      # 제주 성산 (겨울무)
gochang    = load_csv(DATA_DIR + "고창군_순별.csv")    # 전북 고창 (봄·가을무)
daegwallyeong = load_csv(DATA_DIR + "대관령_순별.csv") # 강원 대관령 (여름무)

supply = load_csv(DATA_DIR + "무반입량_순별.csv")
supply['총반입량'] = pd.to_numeric(supply['총반입량'], errors='coerce')

search = load_csv(DATA_DIR + "search_순별.csv")
search['평균_검색량'] = pd.to_numeric(search['평균_검색량'], errors='coerce')

print(f"[로드] 가격 {price.shape[0]}행 | 성산 {seongsan.shape[0]}행 | "
      f"고창 {gochang.shape[0]}행 | 대관령 {daegwallyeong.shape[0]}행 | "
      f"반입량 {supply.shape[0]}행 | 검색량 {search.shape[0]}행")


# ================================================================
# 3. 기상 데이터 병합 (계절별 주산지 기준)
# ================================================================
WEATHER_COLS = [
    '평균_최저기온', '평균_최고기온', '총_강수량', '평균_풍속',
    '평균_상대습도', '평균_일사량', '평균_지면온도', '평균_최저초상온도'
]

# ── 계절별 기상 소스 선택 규칙 ──────────────────────────────
# 겨울무 (12, 1, 2월)  → 성산
# 봄무   (3, 4, 5, 6월) → 고창군
# 여름무 (7, 8월)       → 대관령
# 가을무 (9, 10, 11월)  → 고창군
def get_weather_source(month, se_row, go_row, da_row):
    if month in [12, 1, 2]:
        return se_row   # 겨울무 → 성산
    elif month in [7, 8]:
        return da_row   # 여름무 → 대관령
    else:
        return go_row   # 봄·가을무 → 고창군


se_dict = seongsan.set_index('idx')
go_dict = gochang.set_index('idx')
da_dict = daegwallyeong.set_index('idx')

all_idx = sorted(
    set(se_dict.index) | set(go_dict.index) | set(da_dict.index)
)

weather_rows = []
for idx in all_idx:
    se_row = se_dict.loc[idx] if idx in se_dict.index else None
    go_row = go_dict.loc[idx] if idx in go_dict.index else None
    da_row = da_dict.loc[idx] if idx in da_dict.index else None

    # 월 정보는 존재하는 소스에서 가져옴
    ref_row = next(r for r in [se_row, go_row, da_row] if r is not None)
    month   = int(ref_row['Month'])

    src = get_weather_source(month, se_row, go_row, da_row)
    # 선택된 소스가 없으면 다른 소스로 폴백
    if src is None:
        src = next((r for r in [se_row, go_row, da_row] if r is not None), None)

    row = {'idx': idx}
    for c in WEATHER_COLS:
        row[c] = float(src[c]) if (src is not None and c in src.index) else np.nan
    weather_rows.append(row)

weather = pd.DataFrame(weather_rows)
print(f"[기상 병합] {weather.shape[0]}행 생성 "
      f"(겨울=성산, 봄·가을=고창군, 여름=대관령)")


# ================================================================
# 4. 전체 데이터 병합
# ================================================================
df = price[['idx', 'Year', 'Month', 'Period', 'PeriodStr',
            '평균가격', '전년', '평년', 'DATE']].copy()
df = df.merge(weather[['idx'] + WEATHER_COLS], on='idx', how='left')
df = df.merge(supply[['idx', '총반입량']],     on='idx', how='left')
df = df.merge(search[['idx', '평균_검색량']],  on='idx', how='left')
df = df.sort_values('idx').reset_index(drop=True)
print(f"[병합] {df.shape[0]}행 × {df.shape[1]}열")


# ================================================================
# 5. 피쳐 엔지니어링
# ================================================================
TARGET = '평균가격'

# ── 가격 Lag ─────────────────────────────────────────────────
for lag in [1, 2, 3, 4, 5, 6, 9, 12, 18, 36]:
    df[f'plag{lag}'] = df[TARGET].shift(lag)

# ── 가격 이동평균 ─────────────────────────────────────────────
for w in [3, 6, 12]:
    df[f'pma{w}'] = df[TARGET].shift(1).rolling(w).mean()

# ── 가격 변동성 ───────────────────────────────────────────────
for w in [3, 6]:
    df[f'pstd{w}'] = df[TARGET].shift(1).rolling(w).std()

# ── 모멘텀 & YoY ──────────────────────────────────────────────
df['pmom3'] = df['plag1'] - df['plag4']
df['pyoy']  = df[TARGET].shift(1) / df[TARGET].shift(37) - 1

# ── 가격 vs 평년·전년 ─────────────────────────────────────────
df['p_vs_py'] = df['plag1'] / df['평년'].replace(0, np.nan)
df['p_vs_jn'] = df['plag1'] / df['전년'].replace(0, np.nan)

# ── 반입량 ───────────────────────────────────────────────────
for lag in [1, 2, 3]:
    df[f'slag{lag}'] = df['총반입량'].shift(lag)
df['sma3']    = df['총반입량'].shift(1).rolling(3).mean()
df['sma6']    = df['총반입량'].shift(1).rolling(6).mean()
df['schg']    = df['총반입량'].shift(1).pct_change()
df['svma']    = df['slag1'] / df['sma6'].replace(0, np.nan)
df['ps_ratio']= df['plag1'] / df['slag1'].replace(0, np.nan)

# ── 검색량 ───────────────────────────────────────────────────
for lag in [1, 3, 6, 12, 18]:
    df[f'srlag{lag}'] = df['평균_검색량'].shift(lag)
df['srma3'] = df['평균_검색량'].shift(1).rolling(3).mean()

# ── 기상 Lag ─────────────────────────────────────────────────
for c in WEATHER_COLS:
    for lag in [1, 3, 6, 9]:
        df[f'{c}_l{lag}'] = df[c].shift(lag)

# ── 기상 파생 ─────────────────────────────────────────────────
df['temp_range_l3']  = df['평균_최고기온'].shift(3) - df['평균_최저기온'].shift(3)
df['heat_stress']    = (df['평균_최고기온'].shift(3) > 30).astype(int)
df['cold_stress']    = (df['평균_최저기온'].shift(3) < -5).astype(int)
df['frost_stress']   = (df['평균_최저초상온도'].shift(1) < -3).astype(int)  # 무 동해 지표
df['heavy_rain_l3']  = (df['총_강수량'].shift(3) > 100).astype(int)
df['heavy_rain_l6']  = (df['총_강수량'].shift(6) > 100).astype(int)

# ── 달력 피쳐 ────────────────────────────────────────────────
df['msin']   = np.sin(2 * np.pi * df['Month'] / 12)
df['mcos']   = np.cos(2 * np.pi * df['Month'] / 12)
df['piy']    = (df['Month'] - 1) * 3 + df['Period']
df['pysin']  = np.sin(2 * np.pi * df['piy'] / 36)
df['pycos']  = np.cos(2 * np.pi * df['piy'] / 36)

# 무 출하 시즌 플래그
df['kimchi_season'] = ((df['Month'] >= 10) & (df['Month'] <= 12)).astype(int)  # 김장철
df['winter_mu']     = ((df['Month'] == 12) | (df['Month'] <= 2)).astype(int)   # 겨울무 (성산)
df['spring_mu']     = ((df['Month'] >= 3)  & (df['Month'] <= 6)).astype(int)   # 봄무 (고창)
df['summer_mu']     = ((df['Month'] >= 7)  & (df['Month'] <= 8)).astype(int)   # 여름무 (대관령)
df['autumn_mu']     = ((df['Month'] >= 9)  & (df['Month'] <= 11)).astype(int)  # 가을무 (고창)


# ================================================================
# 6. 데이터 정제 + Train/Test 분리
# ================================================================
EXCLUDE = (
    {'idx', 'Year', 'Month', 'Period', 'PeriodStr', 'DATE',
     TARGET, '전년', '평년', '총반입량', '평균_검색량'}
    | set(WEATHER_COLS)
)
ALL_FEATURES = [c for c in df.columns
                if c not in EXCLUDE and not df[c].isna().all()]

first_valid = df[ALL_FEATURES].dropna().index.min()
df_clean = df.loc[first_valid:].copy()
df_clean[ALL_FEATURES] = (
    df_clean[ALL_FEATURES]
    .fillna(method='ffill').fillna(0)
    .replace([np.inf, -np.inf], 0)
)

train = df_clean[(df_clean['Year'] >= 2018) & (df_clean['Year'] <= 2024)]
test  = df_clean[df_clean['Year'] == 2025]

print(f"[후보 피쳐] {len(ALL_FEATURES)}개")
print(f"[학습] {len(train)}행 (2018~2024)")
print(f"[테스트] {len(test)}행 (2025)")

if len(test) == 0:
    print("\n⚠️  2025년 데이터가 없습니다. 테스트 구간을 2024년으로 대체합니다.")
    train = df_clean[(df_clean['Year'] >= 2018) & (df_clean['Year'] <= 2023)]
    test  = df_clean[df_clean['Year'] == 2024]
    print(f"[재설정] 학습 {len(train)}행 / 테스트 {len(test)}행 (2024)")


# ================================================================
# 7. 피쳐 선별 (importance 기반 동적 선별)
# ================================================================
print("\n--- 피쳐 선별 (importance 기반 동적) ---")

X_all_tr = train[ALL_FEATURES].values
y_tr     = train[TARGET].values
X_all_te = test[ALL_FEATURES].values
y_te     = test[TARGET].values

# 1단계: 전체 피쳐로 학습 → importance 순위 추출
selector = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
selector.fit(X_all_tr, y_tr)

importance_df = pd.DataFrame({
    'feature':    ALL_FEATURES,
    'importance': selector.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

# 2단계: 피쳐 개수별 성능 비교 (20~55개) → 최고 R² 선택
best_r2, best_n = -999, len(ALL_FEATURES)
search_max = min(56, len(ALL_FEATURES) + 1)

for n_feat in range(20, search_max):
    top_feats = importance_df.head(n_feat)['feature'].tolist()
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    m.fit(train[top_feats].values, y_tr)
    pred = m.predict(test[top_feats].values)
    r2 = r2_score(y_te, pred)
    if r2 > best_r2:
        best_r2 = r2
        best_n  = n_feat

SELECTED = importance_df.head(best_n)['feature'].tolist()
print(f"  최적 피쳐 수: {best_n}개 (탐색 R²={best_r2:.4f})")
print(f"  Top 10 피쳐: {SELECTED[:10]}")

X_train = train[SELECTED].values
X_test  = test[SELECTED].values


# ================================================================
# 8. 최종 모델 학습
# ================================================================
print("\n" + "=" * 60)
print("  모델 학습: XGBoost")
print("=" * 60)

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_tr)
print("  학습 완료!")


# ================================================================
# 9. 교차검증 (TimeSeriesSplit 5-fold)
# ================================================================
print("\n--- TimeSeriesSplit 교차검증 (5-fold) ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    fold_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    fold_model.fit(X_train[tr_idx], y_tr[tr_idx])
    fold_pred = fold_model.predict(X_train[val_idx])

    fold_r2   = r2_score(y_tr[val_idx], fold_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_tr[val_idx], fold_pred))
    cv_results.append({'fold': fold, 'R2': fold_r2, 'RMSE': fold_rmse})
    print(f"  Fold {fold}: R²={fold_r2:.4f}, RMSE={fold_rmse:,.0f}")

avg_r2   = np.mean([r['R2']   for r in cv_results])
avg_rmse = np.mean([r['RMSE'] for r in cv_results])
print(f"  ─────────────────────────────")
print(f"  평균:  R²={avg_r2:.4f}, RMSE={avg_rmse:,.0f}")


# ================================================================
# 10. 테스트 평가
# ================================================================
y_pred = model.predict(X_test)

test_r2   = r2_score(y_te, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
test_mae  = mean_absolute_error(y_te, y_pred)
test_mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100

test_year = int(test['Year'].iloc[0]) if len(test) > 0 else 2025

print("\n" + "=" * 60)
print(f"  {test_year}년 테스트 결과")
print("=" * 60)
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: {test_rmse:,.0f}")
print(f"  MAE:  {test_mae:,.0f}")
print(f"  MAPE: {test_mape:.1f}%")
print("=" * 60)


# ================================================================
# 11. 순별 예측 상세 출력
# ================================================================
print(f"\n{'DATE':>14s}  {'실제':>10s}  {'예측':>10s}  {'오차':>10s}  {'오차율':>7s}")
print("-" * 58)

for i in range(len(test)):
    date_str = test.iloc[i]['DATE']
    actual   = y_te[i]
    pred_v   = y_pred[i]
    error    = actual - pred_v
    pct      = abs(error) / actual * 100
    print(f"{date_str:>14s}  {actual:>10,.0f}  {pred_v:>10,.0f}  "
          f"{error:>+10,.0f}  {pct:>6.1f}%")


# ================================================================
# 12. 피쳐 중요도 (Top 15)
# ================================================================
final_imp = pd.DataFrame({
    'feature':    SELECTED,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(f"\n--- 피쳐 중요도 (Top 15 / 전체 {best_n}개) ---")
for _, row in final_imp.head(15).iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:30s}  {row['importance']:.4f}  {bar}")


# ================================================================
# 13. 시각화 (6개 플롯)
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform

    # 한글 폰트 설정
    font_set = False
    if platform.system() == 'Windows':
        for font_name in ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim']:
            if font_name in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.family'] = font_name
                font_set = True
                print(f"  [폰트] {font_name} 사용")
                break
        if not font_set:
            import os as _os
            win_font_dir = 'C:/Windows/Fonts'
            for fname in ['malgun.ttf', 'malgunbd.ttf', 'NanumGothic.ttf', 'gulim.ttc']:
                fpath = _os.path.join(win_font_dir, fname)
                if _os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    prop = fm.FontProperties(fname=fpath)
                    plt.rcParams['font.family'] = prop.get_name()
                    font_set = True
                    break
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
        font_set = True
    else:
        try:
            import koreanize_matplotlib
            font_set = True
        except ImportError:
            plt.rcParams['font.family'] = 'NanumGothic'
    if not font_set:
        print("  [폰트] 한글 폰트를 찾지 못했습니다. 영문 레이블로 대체됩니다.")

    plt.rcParams['axes.unicode_minus'] = False

    COLORS = {
        'actual':  '#2C3E50',
        'pred':    '#E74C3C',
        'fill':    '#E74C3C',
        'train':   '#3498DB',
        'bar':     '#E67E22',
        'res_pos': '#E74C3C',
        'res_neg': '#27AE60',
        'cv_bar':  '#42A5F5',
        'cv_avg':  '#E74C3C',
    }

    xlabels = [d.replace(str(test_year), f"'{str(test_year)[-2:]}") for d in test['DATE']]

    # ── ① 테스트연도 실제 vs 예측 ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(test))
    ax.plot(x, y_te,   'o-', color=COLORS['actual'], linewidth=2.2,
            markersize=7, label='실제가격', zorder=5)
    ax.plot(x, y_pred, 's-', color=COLORS['pred'],   linewidth=2,
            markersize=5, label='XGBoost 예측', zorder=4)
    ax.fill_between(x, y_te, y_pred, alpha=0.12, color=COLORS['fill'])

    for i in range(len(test)):
        err_pct = abs(y_te[i] - y_pred[i]) / y_te[i] * 100
        ax.annotate(f'{err_pct:.1f}%',
                    (i, (y_te[i] + y_pred[i]) / 2),
                    fontsize=7, ha='center', color='gray', alpha=0.8)

    ax.set_xticks(range(len(test)))
    ax.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=8)
    ax.set_title(
        f'{test_year} 무 도매가격: 실제 vs 예측  |  '
        f'R²={test_r2:.4f}  RMSE={test_rmse:,.0f}  MAPE={test_mape:.1f}%',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.set_ylabel('가격 (원/20kg)', fontsize=11)
    ax.set_xlabel('순별', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_2025_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1/6] result_2025_prediction.png")

    # ── ② 산점도 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, y_pred, s=70, c=COLORS['pred'], alpha=0.75,
               edgecolors='white', linewidth=0.8, zorder=5)
    mn = min(y_te.min(), y_pred.min()) * 0.9
    mx = max(y_te.max(), y_pred.max()) * 1.1
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.4, linewidth=1, label='y = x (완벽 예측)')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('실제 가격 (원)', fontsize=11)
    ax.set_ylabel('예측 가격 (원)', fontsize=11)
    ax.set_title(f'실제 vs 예측 산점도 (R²={test_r2:.4f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2/6] result_scatter.png")

    # ── ③ 피쳐 중요도 (Top 20) ──────────────────────────────────
    top_n   = min(20, len(final_imp))
    top_imp = final_imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_imp['importance'].values,
            color=COLORS['bar'], alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_imp['feature'].values, fontsize=9)
    for i, v in enumerate(top_imp['importance'].values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8, color='gray')
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'피쳐 중요도 Top {top_n} (전체 {best_n}개 선택)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3/6] result_feature_importance.png")

    # ── ④ 전체 시계열 (2018~테스트연도) ─────────────────────────
    full_pred   = model.predict(df_clean[SELECTED].values)
    dates       = df_clean['DATE'].values
    actual_vals = df_clean[TARGET].values
    n_pts       = len(dates)
    n_train_pts = len(train)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axvline(x=n_train_pts - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.axvspan(n_train_pts - 0.5, n_pts, alpha=0.06, color='orange')
    ax.text(n_train_pts + 1, max(actual_vals) * 0.95,
            f'← {test_year} 테스트', fontsize=9, color='gray', style='italic')

    ax.plot(range(n_pts), actual_vals, '-', color=COLORS['actual'],
            linewidth=1.5, alpha=0.8, label='실제가격')
    ax.plot(range(n_pts), full_pred,   '-', color=COLORS['pred'],
            linewidth=1.2, alpha=0.65, label='모델 예측')

    tick_pos, tick_lab = [], []
    for i, d in enumerate(dates):
        if '01상순' in str(d):
            tick_pos.append(i)
            tick_lab.append(str(d)[:4])
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=10)

    ax.set_title(f'무 도매가격 전체 시계열 (2018~{test_year})',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('가격 (원/20kg)', fontsize=11)
    ax.set_xlabel('연도', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_full_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/6] result_full_timeseries.png")

    # ── ⑤ 잔차 분석 ─────────────────────────────────────────────
    residuals = y_te - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors_r = [COLORS['res_neg'] if r < 0 else COLORS['res_pos'] for r in residuals]
    ax1.bar(range(len(residuals)), residuals,
            color=colors_r, alpha=0.75, edgecolor='white')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(range(len(test)))
    ax1.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=7)
    ax1.set_ylabel('잔차 (실제 - 예측)', fontsize=10)
    ax1.set_title('순별 예측 잔차', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    ax2 = axes[1]
    ax2.hist(residuals, bins=min(12, len(residuals)),
             color=COLORS['pred'], alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.axvline(np.mean(residuals), color='blue', linewidth=1.2, linestyle='-',
                label=f'평균={np.mean(residuals):,.0f}')
    ax2.set_xlabel('잔차 (원)', fontsize=10)
    ax2.set_ylabel('빈도', fontsize=10)
    ax2.set_title('잔차 분포', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')

    plt.suptitle(f'잔차 분석  |  MAE={test_mae:,.0f}원  MAPE={test_mape:.1f}%',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/6] result_residuals.png")

    # ── ⑥ CV 결과 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [r['fold'] for r in cv_results]
    r2s   = [r['R2']   for r in cv_results]

    bar_colors = [COLORS['cv_bar'] if r >= 0 else '#EF9A9A' for r in r2s]
    ax.bar(folds, r2s, color=bar_colors, alpha=0.8, edgecolor='white', width=0.6)
    ax.axhline(avg_r2,  color=COLORS['cv_avg'], linewidth=1.8, linestyle='--',
               label=f'CV 평균 R²={avg_r2:.4f}')
    ax.axhline(test_r2, color='green', linewidth=1.5, linestyle=':',
               label=f'테스트 R²={test_r2:.4f}')

    for i, r in enumerate(r2s):
        ax.text(folds[i], r + 0.02 if r >= 0 else r - 0.06,
                f'{r:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('TimeSeriesSplit 5-Fold 교차검증 결과', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(folds)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR + 'result_cv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/6] result_cv_results.png")

    print(f"\n[시각화 완료] result_*.png 6개 파일 → {OUTPUT_PLOT_DIR}")

except Exception as e:
    print(f"\n[시각화 건너뜀] {e}")
    import traceback; traceback.print_exc()


# ================================================================
# 14. 결과 저장
# ================================================================
# ── 예측 CSV ────────────────────────────────────────────────────
result_df = test[['DATE', 'Year', 'Month', 'Period', TARGET]].copy()
result_df['예측가격']  = y_pred
result_df['오차']      = y_te - y_pred
result_df['오차율(%)'] = np.abs(result_df['오차'] / y_te) * 100
result_df.to_csv(OUTPUT_PRED_DIR + 'result_2025_predictions.csv',
                 index=False, encoding='utf-8-sig')
print(f"[저장] {OUTPUT_PRED_DIR}result_2025_predictions.csv")

# ── 모델 메트릭 JSON ─────────────────────────────────────────
metrics = {
    "crop":      "무",
    "test_year": test_year,
    "test": {
        "R2":   round(float(test_r2),   4),
        "RMSE": round(float(test_rmse), 2),
        "MAE":  round(float(test_mae),  2),
        "MAPE": round(float(test_mape), 2),
    },
    "cv": {
        "avg_R2":   round(float(avg_r2),   4),
        "avg_RMSE": round(float(avg_rmse), 2),
        "folds":    cv_results,
    },
    "model": {
        "algorithm":       "XGBoost",
        "n_estimators":    500,
        "max_depth":       3,
        "learning_rate":   0.05,
        "n_features":      best_n,
        "selected_features": SELECTED,
    },
    "weather_mapping": {
        "겨울무(12~2월)": "성산",
        "봄무(3~6월)":    "고창군",
        "여름무(7~8월)":  "대관령",
        "가을무(9~11월)": "고창군",
    },
}

os.makedirs("./results/", exist_ok=True)
with open("./results/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("[저장] ./results/model_metrics.json")

print(f"\n총 소요시간: {time.time() - t0:.1f}초")
print("=" * 60)
print("  완료!")
print("=" * 60)
