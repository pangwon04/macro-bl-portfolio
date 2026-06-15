# -*- coding: utf-8 -*-
"""
Created on Wed May 20 05:49:41 2026

@author: 윤창현
"""

# -*- coding: utf-8 -*-
"""
=============================================================
Macro-Driven Black-Litterman Portfolio Optimization Pipeline
Version 4.3
=============================================================

변경 사항 (v4.2 → v4.3):
──────────────────────────────────────────────────────────────
[1] R² 기반 자동 공분산 블렌딩
    ▸ 역사적 EWMA 공분산 + 팩터 β 기반 공분산을 자산별 R²로 자동 가중
    ▸ W_ij = (R²[i] + R²[j]) / 2
    ▸ Σ_blend[i,j] = W_ij × Σ_factor[i,j] + (1-W_ij) × Σ_EWMA[i,j]
    ▸ 팩터 공분산:
        Σ_factor = B × Σ_F × B' + D
        B: (N자산 × K팩터) 베타 행렬
        Σ_F: 팩터 수익률 공분산 (K × K)
        D: 잔차 분산 대각행렬 (N × N)

[2] v4.2 기능 전부 유지
    ▸ SCENARIO_PROBS 직접 입력
    ▸ DNS v4.2 팩터 보정 (Level 분해, L+S BEI 차감)
    ▸ λ 최적화 + VAR 정상성 강제
    ▸ 블렌딩 포트폴리오 (Risk-On / Risk-Off / 블렌딩)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
from scipy import stats
import subprocess, sys


# ============================================================
# ■ 한글 폰트 설정
# ============================================================
def setup_korean_font():
    """
    matplotlib 한글 폰트 자동 설정.
    우선순위: 시스템 설치 폰트 → 경로 직접 탐색 → koreanize-matplotlib 설치
    """
    candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "Apple SD Gothic Neo", "AppleGothic",
        "Noto Sans KR", "Noto Sans CJK KR",
        "Gulim", "Dotum", "Batang",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  ✓ 한글 폰트: {name}")
            return

    # 폰트 파일 직접 탐색
    search_dirs = [
        "/usr/share/fonts", "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        "/System/Library/Fonts", "/Library/Fonts",
        "C:/Windows/Fonts",
        "C:/Users/" + os.environ.get("USERNAME","") + "/AppData/Local/Microsoft/Windows/Fonts",
    ]
    keywords = ["Nanum", "nanum", "Gothic", "gothic", "Malgun", "malgun",
                "gulim", "Gulim", "dotum", "Dotum"]
    for d in search_dirs:
        if not os.path.isdir(d): continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".ttf") and any(k in f for k in keywords):
                    path = os.path.join(root, f)
                    try:
                        fm.fontManager.addfont(path)
                        name = fm.FontProperties(fname=path).get_name()
                        plt.rcParams["font.family"] = name
                        plt.rcParams["axes.unicode_minus"] = False
                        print(f"  ✓ 한글 폰트 (직접 로드): {name}  [{path}]")
                        return
                    except Exception:
                        pass

    # koreanize-matplotlib 설치 시도
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "koreanize-matplotlib", "-q"])
        import koreanize_matplotlib   # noqa: F401
        plt.rcParams["axes.unicode_minus"] = False
        print("  ✓ koreanize-matplotlib 설치 완료")
        return
    except Exception:
        pass

    # 최종 fallback: DejaVu Sans (한글 깨짐 허용)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    print("  ⚠ 한글 폰트 없음 → DejaVu Sans 사용 (한글 깨짐 가능)")
    print("    해결: pip install koreanize-matplotlib  또는 나눔폰트 설치")


setup_korean_font()

try:
    import yfinance as yf
    _YFINANCE_OK = True
except ImportError:
    _YFINANCE_OK = False
    print("⚠ yfinance 미설치 → pip install yfinance")

try:
    import pandas_datareader.data as web
    _PDR_OK = True
except Exception:
    _PDR_OK = False
    print("⚠ pandas-datareader 미설치 → pip install pandas-datareader")

try:
    from fredapi import Fred as _FredApi
    _FREDAPI_OK = True
except ImportError:
    _FREDAPI_OK = False

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


# ============================================================
# ■ GLOBAL CONFIG
# ============================================================
OUTPUT_DIR            = os.getenv("OUTPUT_DIR", "outputs")
START_DATE            = "2005-01-01"
END_DATE              = datetime.today().strftime("%Y-%m-%d")
TOTAL_PORTFOLIO_VALUE = 100_000
RISK_FREE_RATE        = 0.045
TAU                   = 0.05
HIST_LOOKBACK         = 120
USE_EWMA_COV          = True
EWMA_SPAN             = 500
USE_3M_REGRESSION     = True
REGRESSION_METHOD     = "OLS_HAC"
FRED_API_KEY          = os.getenv("FRED_API_KEY", "")       # .env 또는 환경변수에서 입력

# ── 비중 제약 ─────────────────────────────────────────────────
MIN_WEIGHT_ALL  = 0.05
USE_MDD_WMAX    = True
MDD_LOOKBACK    = None
MDD_BASE_MAX    = 0.30
MDD_WMAX_FLOOR  = 0.05
MDD_WMAX_CAP    = 0.40
BOND_TOTAL_MIN  = 0.20

SCENARIO_HORIZONS = [3, 6]

# ─────────────────────────────────────────────────────────────
# 시나리오 확률 직접 입력
# ─────────────────────────────────────────────────────────────
SCENARIO_PROBS = {
    "Risk-On":  0.60,
    "Risk-Off": 0.40,
}

# ─────────────────────────────────────────────────────────────
# [v4.4] R² 기반 자동 블렌딩
# ─────────────────────────────────────────────────────────────
# 수익률 Prior:
#   Prior[i] = R²[i] × mu_factor[i] + (1-R²[i]) × mu_hist[i]
#            = mu_hist[i] + R²[i] × scenario_alpha[i]
#
# 공분산:
#   W_ij = (R²[i] + R²[j]) / 2
#   Σ_blend[i,j] = W_ij × Σ_factor[i,j] + (1-W_ij) × Σ_EWMA[i,j]
#
# R²가 높은 자산 → 팩터 설명력 높음 → 팩터 기반 추정 더 신뢰
# R²가 낮은 자산 → 역사적 수익률/변동성 더 신뢰
#
# USE_FACTOR_COV    = False: 공분산 순수 EWMA만 사용
# USE_FACTOR_RETURN = False: 수익률 순수 mu_hist만 사용
# ─────────────────────────────────────────────────────────────
USE_FACTOR_COV    = True   # False: 순수 EWMA
USE_FACTOR_RETURN = True   # False: 순수 mu_hist

# ─────────────────────────────────────────────────────────────
# [6M 기준] 최적화 기간 통일
# ─────────────────────────────────────────────────────────────
# 기대수익률·공분산·무위험수익률을 모두 6개월 기준으로 맞춰
# 단위 불일치 문제를 제거한다.
#
# mu_hist_6m    = 월간 평균수익률 × 6
# alpha_6m      = 3M 회귀 alpha  × 2  (3M → 6M)
# Σ_6m          = Σ_annual × (6/12)
# rf_6m         = rf_annual × (6/12)
#
# 성과 표시는 최적화 결과를 역산해 연환산:
#   r_ann   = (1 + r_6m)^(12/6) - 1 = (1 + r_6m)^2 - 1
#   σ_ann   = σ_6m × √(12/6) = σ_6m × √2
#   Sharpe  = (r_ann - rf_ann) / σ_ann
# ─────────────────────────────────────────────────────────────
HORIZON_MONTHS    = 6                              # 최적화 기간 (개월)
RISK_FREE_RATE_6M = RISK_FREE_RATE * HORIZON_MONTHS / 12  # 2.25%

# 팩터 시차
FACTOR_LAGS = {
    "Inflation Factor":  0,
    "Real Rate Factor":  0,
    "Credit Factor":     0,
    "Growth Factor":     1,
    "Labor Factor":      1,
    "Commodity Factor":  0,
}

# DNS 파라미터
DNS_VAR_MAX_RHO = 0.95


# ============================================================
# ■ STEP 1: ASSET UNIVERSE
# ============================================================
ASSETS = {
    "SHY":  {"name": "단기 국채 (1-3Y)",   "type": "bond_short"},
    "IEF":  {"name": "중기 국채 (7-10Y)",  "type": "bond_mid"},
    "LQD":  {"name": "IG 회사채",          "type": "bond_credit"},
    "QQQ":  {"name": "나스닥-100",         "type": "equity_growth"},
    "SPHQ": {"name": "S&P 500 Quality",   "type": "equity_quality"},
    "XLI":  {"name": "산업재 섹터",        "type": "equity_infra"},
    "XLP":  {"name": "필수 소비재",        "type": "equity_staples"},
    "GLD":  {"name": "금",                "type": "commodity_gold"},
    "USO":  {"name": "WTI 원유 선물",      "type": "commodity_oil"},
    "DBA":  {"name": "농산물 원자재",      "type": "commodity_agri"},
}
PORTFOLIO_ASSETS = list(ASSETS.keys())

#USER_BL_VIEWS = {
 #   "Risk-On":  {},
 #   "Risk-Off": {},
#}
# annual_return이지만 6개월 단위
USER_BL_VIEWS = {
    "Risk-On": {
        # 완화적 금융환경 + AI·제조업 투자 확산
        # 성장주와 산업재를 중심으로 두되, Prior와 크게 괴리되지 않게 설정
        "QQQ":  {"annual_return": 0.115, "confidence": 0.60},
        "XLI":  {"annual_return": 0.105, "confidence": 0.60},

        # 퀄리티·필수소비재는 Risk-On에서도 포트폴리오 안정성 보완
        "SPHQ": {"annual_return": 0.085, "confidence": 0.60},
        "XLP":  {"annual_return": 0.050, "confidence": 0.60},

        # 금리 안정·신용스프레드 축소 수혜는 있으나 회사채 상승 여력은 제한
        "LQD":  {"annual_return": 0.042, "confidence": 0.60},

        # Risk-On에서는 방어 프리미엄이 약해지므로 대체자산은 Prior보다 약간 낮게
        "GLD":  {"annual_return": 0.060, "confidence": 0.60},
        "USO":  {"annual_return": 0.065, "confidence": 0.60},
        "DBA":  {"annual_return": 0.030, "confidence": 0.60},
    },

    "Risk-Off": {
        # 단기채: 듀레이션 리스크가 작고, 안전자산·현금성 수요 증가 반영
        "SHY":  {"annual_return": 0.015, "confidence": 0.90},

        # 중기국채: 안전자산 수요는 반영하되, 실질금리·기대인플레이션 상승 부담 때문에 낮게 제한
        "IEF":  {"annual_return": 0.005, "confidence": 0.90},

        # IG 회사채: 신용스프레드 확대 국면에서는 방어자산으로 보기 어려움
        "LQD":  {"annual_return": -0.015, "confidence": 0.60},

        # 성장주·산업재: 할인율 상승, 경기둔화, 위험회피에 취약
        "QQQ":  {"annual_return": 0.000, "confidence": 0.60},
        "XLI":  {"annual_return": 0.000, "confidence": 0.60},

        # 퀄리티·필수소비재: 주식 내 방어적 익스포저 유지
        "SPHQ": {"annual_return": 0.040, "confidence": 0.60},
        "XLP":  {"annual_return": 0.040, "confidence": 0.60},

        # 금·원자재: 재인플레이션 방어 역할은 인정하되, 변동성·꼬리위험 때문에 과도하게 높이지 않음
        "GLD":  {"annual_return": 0.050, "confidence": 0.60},
        "USO":  {"annual_return": 0.075, "confidence": 0.60},
        "DBA":  {"annual_return": 0.050, "confidence": 0.60},
    },
}
SCENARIOS = {
    "Risk-On": {
        "description": "완화적 금융환경 + AI·제조업 투자 확산",
        "prob": SCENARIO_PROBS["Risk-On"],
        "color": "#2ecc71",
        "deltas": {
            "Inflation Factor":  -0.4,
            "Real Rate Factor":  -0.3,
            "Credit Factor":     -0.5,
            "Growth Factor":     +1.0,
            "Labor Factor":      -0.3,
            "Commodity Factor":  +0.1,
        },
    },
    "Risk-Off": {
        "description": "재인플레이션 + 경기 둔화 — Stagflation",
        "prob": SCENARIO_PROBS["Risk-Off"],
        "color": "#e74c3c",
        "deltas": {
            "Inflation Factor":  +0.8,
            "Real Rate Factor":  +0.8,
            "Credit Factor":     +1.3,
            "Growth Factor":     -1.0,
            "Labor Factor":      +0.5,
            "Commodity Factor":  +0.8,
        },
    },
}
SC_COLORS = {sc: cfg["color"] for sc, cfg in SCENARIOS.items()}


# ============================================================
# ■ HELPER UTILITIES
# ============================================================
def _monthly_end(start, end):
    try:   return pd.date_range(start, end, freq="ME")
    except: return pd.date_range(start, end, freq="M")

def _fred(series_code, start, end):
    key = FRED_API_KEY.strip()
    if _FREDAPI_OK and key:
        return _FredApi(api_key=key).get_series(
            series_code, observation_start=start, observation_end=end)
    if _PDR_OK:
        if key:
            from pandas_datareader.fred import FredReader
            return FredReader(series_code, start=start, end=end,
                              api_key=key).read().iloc[:, 0]
        return web.DataReader(series_code, "fred", start, end).iloc[:, 0]
    raise RuntimeError("FRED 접근 불가")

def _to_monthly_end(series):
    s = series.copy()
    try:
        ym = pd.to_datetime(s.index).to_period("M").strftime("%Y-%m")
        tmp = pd.DataFrame({"v": s.values}, index=ym)
        tmp = tmp[~tmp.index.duplicated(keep="last")]
        new_idx = pd.to_datetime([f"{y}-01" for y in tmp.index]) + pd.offsets.MonthEnd(0)
        s = pd.Series(tmp["v"].values, index=new_idx, name=s.name)
    except Exception: pass
    return s

def _resample_monthly(raw):
    try:   return raw.resample("ME").last().ffill()
    except: return raw.resample("M").last().ffill()

def _zscore(series):
    mu, sig = series.mean(), series.std()
    if sig == 0 or np.isnan(sig): return series
    return (series - mu) / sig

def _make_synthetic_prices(tickers, start, end, seed=42):
    params = {
        "SHY": (0.035,0.015), "IEF": (0.040,0.060), "LQD": (0.050,0.075),
        "QQQ": (0.140,0.210), "SPHQ":(0.110,0.165), "XLI": (0.105,0.175),
        "XLP": (0.080,0.130), "GLD": (0.060,0.155), "USO": (0.010,0.400),
        "DBA": (0.020,0.185),
    }
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    tickers = [t for t in tickers if t in params]
    k = len(tickers); idx = {t: i for i, t in enumerate(tickers)}
    corr = np.eye(k)
    for grp, rho in [(["QQQ","SPHQ","XLI"],0.75),
                     (["SHY","IEF","LQD"],  0.80),
                     (["GLD","USO","DBA"],   0.40)]:
        for a in grp:
            for b in grp:
                if a != b and a in idx and b in idx:
                    corr[idx[a],idx[b]] = rho
    for c in ["QQQ","SPHQ","XLI"]:
        if "XLP" in idx and c in idx: corr[idx["XLP"],idx[c]] = 0.35
    for e in ["QQQ","SPHQ","XLI","XLP"]:
        for d in ["SHY","IEF","LQD"]:
            if e in idx and d in idx: corr[idx[e],idx[d]] = -0.25
    for d in ["SHY","IEF","LQD"]:
        if "XLP" in idx and d in idx: corr[idx["XLP"],idx[d]] = 0.20
    if "GLD" in idx:
        for d in ["SHY","IEF","LQD"]:
            if d in idx: corr[idx["GLD"],idx[d]] = 0.15
    dt    = 1/252
    vols  = np.array([params[t][1] for t in tickers])
    drift = np.array([params[t][0] - 0.5*params[t][1]**2 for t in tickers]) * dt
    cov   = np.outer(vols,vols) * corr * dt
    L_ch  = np.linalg.cholesky(cov + np.eye(k)*1e-10)
    rets  = drift + rng.standard_normal((len(dates),k)) @ L_ch.T
    return pd.DataFrame(100*np.exp(np.cumsum(rets,axis=0)),
                        index=dates, columns=tickers)

def _ensure_psd(M, eps=1e-8):
    """양정치(Positive Semi-Definite) 보장 — 음의 고유값 제거"""
    M = (M + M.T) / 2
    min_eig = np.min(np.linalg.eigvals(M).real)
    if min_eig < eps:
        M += (abs(min_eig) + eps) * np.eye(M.shape[0])
    return M


# ============================================================
# ■ STEP 2: ASSET PRICE DATA
# ============================================================
def fetch_asset_data():
    print("\n[Step 2] 자산 가격 수집...")
    raw = None
    if _YFINANCE_OK:
        try:
            raw = yf.download(PORTFOLIO_ASSETS, start=START_DATE,
                              end=END_DATE, auto_adjust=True,
                              progress=False)["Close"]
            raw = raw.dropna(how="all")
            raw = raw[[t for t in PORTFOLIO_ASSETS if t in raw.columns]]
            if raw.shape[1] >= 3:
                print(f"  ✓ yfinance: {raw.shape[0]}일 × {raw.shape[1]}개 자산")
            else:
                raise ValueError("자산 수 부족")
        except Exception as e:
            print(f"  ⚠ yfinance 실패({e}) → 합성"); raw = None
    if raw is None:
        raw = _make_synthetic_prices(PORTFOLIO_ASSETS, START_DATE, END_DATE)
        print("  ✓ 합성 가격 생성")
    return raw


# ============================================================
# ■ STEP 3: MACRO FACTOR CONSTRUCTION
# ============================================================
def build_inflation_factor(start, end):
    print("  [Inflation] BEI 10Y Δ...")
    try:
        f = _resample_monthly(_fred("T10YIE", start, end)).diff().dropna()
        f.name = "Inflation Factor"; print(f"    ✓ {len(f)}개월"); return f
    except Exception as e:
        print(f"    ⚠ ({e}) → 합성")
    return pd.Series(np.random.default_rng(20).normal(0,0.12,len(_monthly_end(start,end))),
                     index=_monthly_end(start,end), name="Inflation Factor")

def build_real_rate_factor(start, end):
    print("  [Real Rate] DFII10 Δ...")
    for code in ["DFII10","WFII10"]:
        try:
            f = _resample_monthly(_fred(code, start, end)).diff().dropna()
            f.name = "Real Rate Factor"; print(f"    ✓ {code}: {len(f)}개월"); return f
        except Exception as e: print(f"    ⚠ {code}({e})")
    try:
        f = _resample_monthly((_fred("DGS10",start,end)-_fred("T10YIE",start,end)).dropna()).diff().dropna()
        f.name = "Real Rate Factor"; return f
    except Exception: pass
    return pd.Series(np.random.default_rng(40).normal(0,0.15,len(_monthly_end(start,end))),
                     index=_monthly_end(start,end), name="Real Rate Factor")

def build_credit_factor(start, end):
    MIN_MONTHS = 120
    print("  [Credit] 신용 스프레드 Δ (BAA10Y 우선)...")
    for code in ["BAA10Y","BAMLH0A0HYM2","BAMLC0A0CM"]:
        try:
            raw_m = _resample_monthly(_fred(code, start, end))
            f = raw_m.diff().dropna(); f.name = "Credit Factor"
            if len(f) < MIN_MONTHS:
                print(f"    △ {code}: {len(f)}개월 < {MIN_MONTHS} → skip"); continue
            print(f"    ✓ {code}: {len(f)}개월"); return f
        except Exception as e: print(f"    ⚠ {code}({e})")
    return pd.Series(np.random.default_rng(30).normal(0,0.2,len(_monthly_end(start,end))),
                     index=_monthly_end(start,end), name="Credit Factor")

def build_growth_factor(start, end):
    print("  [Growth] INDPRO MoM%...")
    try:
        f = _resample_monthly(_fred("INDPRO", start, end)).pct_change().dropna()*100
        f.name = "Growth Factor"; print(f"    ✓ {len(f)}개월"); return f
    except Exception as e: print(f"    ⚠ ({e}) → 합성 AR(1)")
    rng = np.random.default_rng(50); idx = _monthly_end(start, end); n = len(idx)
    ar = np.zeros(n)
    for i in range(1, n): ar[i] = 0.4*ar[i-1] + rng.normal(0,0.8)
    return pd.Series(ar, index=idx, name="Growth Factor")

def build_labor_factor(start, end):
    print("  [Labor] UNRATE Δ...")
    try:
        f = _resample_monthly(_fred("UNRATE", start, end)).diff().dropna()
        f.name = "Labor Factor"; print(f"    ✓ {len(f)}개월"); return f
    except Exception as e: print(f"    ⚠ ({e}) → 합성")
    return pd.Series(np.random.default_rng(60).normal(0,0.15,len(_monthly_end(start,end))),
                     index=_monthly_end(start,end), name="Labor Factor")

def build_commodity_factor(start, end):
    print("  [Commodity] 원자재 ETF (DJP 우선)...")
    if _YFINANCE_OK:
        for ticker in ["DJP","PDBC","DBA","GSG"]:
            try:
                raw = yf.download(ticker, start=start, end=end,
                                  auto_adjust=True, progress=False)
                px = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw)
                f = _resample_monthly(px.squeeze()).pct_change().dropna()*100
                f.name = "Commodity Factor"
                if f.notna().sum() >= 30:
                    print(f"    ✓ {ticker}: {f.notna().sum()}개월"); return f
            except Exception as e: print(f"    ⚠ {ticker}({e})")
    return pd.Series(np.random.default_rng(70).normal(0,4.0,len(_monthly_end(start,end))),
                     index=_monthly_end(start,end), name="Commodity Factor")

def fetch_macro_data():
    print("\n[Step 3] 매크로 6팩터 수집...")
    factors = [build_inflation_factor(START_DATE, END_DATE),
               build_real_rate_factor(START_DATE, END_DATE),
               build_credit_factor(START_DATE, END_DATE),
               build_growth_factor(START_DATE, END_DATE),
               build_labor_factor(START_DATE, END_DATE),
               build_commodity_factor(START_DATE, END_DATE)]
    factors = [_to_monthly_end(f) for f in factors]
    raw = pd.concat(factors, axis=1, join="outer").dropna()
    if len(raw) == 0:
        idx = _monthly_end(START_DATE, END_DATE)
        cols = ["Inflation Factor","Real Rate Factor","Credit Factor",
                "Growth Factor","Labor Factor","Commodity Factor"]
        raw = pd.DataFrame(np.random.default_rng(99).standard_normal((len(idx),6)),
                           index=idx, columns=cols)
    macro_df = raw.apply(_zscore)
    print(f"\n  기간: {macro_df.index[0].date()}~{macro_df.index[-1].date()}  ({len(macro_df)}개월)")
    for col in macro_df.columns:
        print(f"  {col:22s}: 현재={macro_df[col].iloc[-1]:+.3f}σ")
    return macro_df


# ============================================================
# ■ STEP 4: FACTOR REGRESSION
# ============================================================
def _prepare_returns(prices, macro_df):
    try:   mo = prices.resample("ME").last()
    except: mo = prices.resample("M").last()
    ret1m = mo.pct_change().dropna() * 100
    if USE_3M_REGRESSION:
        ret3m    = mo.pct_change(3).dropna() * 100
        means_3m = ret3m.mean()
        dm_ret3m = ret3m - means_3m
        print(f"  회귀 타겟: 3M demeaned  ({ret3m.index[0].date()}~{ret3m.index[-1].date()})")
        return dm_ret3m, means_3m, ret1m
    return ret1m, ret1m.mean(), ret1m

def _fit_ols_hac(y, X):
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return {"model": model, "betas": model.params, "pvalues": model.pvalues,
            "r2": model.rsquared, "adj_r2": model.rsquared_adj,
            "dw": durbin_watson(model.resid), "resid": model.resid,
            "n_obs": len(y), "f_pval": model.f_pvalue, "method": "OLS_HAC"}

def _fit_robust(y, X):
    rlm   = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    y_hat = X @ rlm.params
    ss_res = np.sum((y.values-y_hat)**2); ss_tot = np.sum((y.values-y.values.mean())**2)
    r2 = max(0., 1-ss_res/ss_tot) if ss_tot>0 else 0.
    n, k = len(y), X.shape[1]-1
    return {"model": rlm, "betas": rlm.params, "pvalues": rlm.pvalues,
            "r2": r2, "adj_r2": 1-(1-r2)*(n-1)/max(n-k-1,1),
            "dw": durbin_watson(y.values-y_hat),
            "resid": pd.Series(y.values-y_hat, index=y.index),
            "n_obs": len(y), "f_pval": np.nan, "method": "ROBUST"}

def run_regressions(reg_returns, macro_df):
    method = "OLS_HAC" if USE_3M_REGRESSION else REGRESSION_METHOD
    print(f"\n[Step 4] 팩터 회귀 ({method}, 6팩터)...")
    X_fixed = pd.DataFrame(index=macro_df.index)
    for factor_name, lag in FACTOR_LAGS.items():
        if factor_name in macro_df.columns:
            X_fixed[f"{factor_name}_L{lag}"] = macro_df[factor_name].shift(lag)
    results = {}
    for asset in PORTFOLIO_ASSETS:
        if asset not in reg_returns.columns: continue
        y = reg_returns[asset].dropna()
        combined = pd.concat([y, X_fixed], axis=1).dropna()
        if len(combined) < 30:
            print(f"  {asset:>5s} │ ⚠ 데이터 부족"); continue
        y_reg = combined.iloc[:,0]; X_reg = sm.add_constant(combined.iloc[:,1:])
        try:
            fit_fn = {"OLS_HAC": _fit_ols_hac, "ROBUST": _fit_robust}.get(method, _fit_ols_hac)
            res = fit_fn(y_reg, X_reg)
            flag = "✓" if res["r2"]>0.20 else ("△" if res["r2"]>0.10 else "✗")
            print(f"  {asset:>5s} │ R²={res['r2']:.3f}{flag} │ adj-R²={res['adj_r2']:.3f} │ N={res['n_obs']}")
            results[asset] = res
        except Exception as e:
            print(f"  {asset:>5s} │ ⚠ 실패({e})")
    return results

def run_regression_diagnostics(regression_results, macro_df):
    print("\n[Step 4b] 회귀 진단...")
    vif_df = pd.DataFrame()
    try:
        X_v = sm.add_constant(macro_df.dropna())
        vif_vals = [variance_inflation_factor(X_v.values, i) for i in range(1, X_v.shape[1])]
        vif_df = pd.DataFrame({"Factor": list(macro_df.columns), "VIF": vif_vals})
        for _, row in vif_df.iterrows():
            flag = "✓" if row.VIF<5 else ("△" if row.VIF<10 else "✗ SEVERE")
            print(f"    {flag} {row.Factor:25s}: {row.VIF:.3f}")
    except Exception as e: print(f"  ⚠ VIF({e})")
    records = []
    for asset, res in regression_results.items():
        jb_p = bp_p = np.nan
        try: _, jb_p = stats.jarque_bera(res["resid"])
        except: pass
        try: _, bp_p, _, _ = het_breuschpagan(res["resid"], res["model"].model.exog)
        except: pass
        records.append({"Asset": asset, "R2": res["r2"], "adj_R2": res["adj_r2"],
                        "DW": res["dw"], "N": res["n_obs"]})
    diag = pd.DataFrame(records).set_index("Asset")
    print("\n  진단 요약:"); print(diag.round(4).to_string())
    return diag, vif_df


# ============================================================
# ■ STEP 5: DNS + KALMAN FILTER (v4.2 유지)
# ============================================================
def _ns_loadings(maturities, lam=0.0609):
    tau = np.asarray(maturities, dtype=float)
    f2  = (1-np.exp(-lam*tau)) / (lam*tau)
    f3  = f2 - np.exp(-lam*tau)
    return np.column_stack([np.ones_like(tau), f2, f3])

def _ns_yield_from_factors(maturities, L, S, C, lam=0.0609):
    lr = _ns_loadings([maturities] if np.isscalar(maturities) else maturities, lam)
    return (float((lr@np.array([L,S,C]))[0]) if np.isscalar(maturities)
            else lr@np.array([L,S,C]))

def fetch_treasury_yields():
    yield_codes = {"DGS3MO":0.25,"DGS6MO":0.5,"DGS1":1,
                   "DGS2":2,"DGS5":5,"DGS10":10,"DGS30":30}
    mats = sorted(yield_codes.values())
    try:
        frames = {m: _fred(c, START_DATE, END_DATE) for c,m in yield_codes.items()}
        df = _resample_monthly(pd.DataFrame(frames)).dropna()
        print(f"  ✓ 국채 금리: {df.shape[0]}개월"); return df, mats
    except Exception as e: print(f"  ⚠ ({e}) → 합성")
    rng = np.random.default_rng(99); idx = _monthly_end(START_DATE, END_DATE)
    n, t = len(idx), np.arange(len(idx))
    base = 2.5+2.5*np.maximum(0,np.sin((t-n*.6)/18))+rng.normal(0,.15,n)
    frames = {m: np.clip(base+.08*(m-10)+rng.normal(0,.05,n),.05,8.) for m in mats}
    return pd.DataFrame(frames, index=idx), mats

def optimize_lambda(yields_df, maturities):
    tau = np.asarray(maturities, dtype=float)
    Y = yields_df.dropna().values
    best_lam = 0.0609; best_r2 = -np.inf
    for lam in np.linspace(0.01, 1.50, 150):
        try:
            Lam = _ns_loadings(tau, lam); r2_sum = 0.0; count = 0
            for row in Y:
                if np.any(np.isnan(row)): continue
                beta = np.linalg.lstsq(Lam, row, rcond=None)[0]
                fitted = Lam @ beta
                ss_res = np.sum((row-fitted)**2); ss_tot = np.sum((row-row.mean())**2)
                if ss_tot > 1e-10: r2_sum += 1.0-ss_res/ss_tot; count += 1
            if count > 0:
                avg_r2 = r2_sum/count
                if avg_r2 > best_r2: best_r2 = avg_r2; best_lam = lam
        except Exception: pass
    return float(best_lam), float(best_r2)

def _make_stationary(Phi, max_rho=DNS_VAR_MAX_RHO):
    eigvals = np.linalg.eigvals(Phi)
    rho = float(np.max(np.abs(eigvals)))
    if rho > max_rho:
        Phi = Phi * (max_rho/rho)
        print(f"    VAR 고유값 {rho:.4f} → {max_rho:.4f} (평균회귀 강제)")
    else:
        print(f"    VAR 고유값 {rho:.4f} ✓")
    return Phi

def dns_kalman_forecast(yields_df, maturities, steps=12):
    print("\n[Step 5] DNS + Kalman Filter (v4.2)...")
    Y = yields_df.replace([np.inf,-np.inf], np.nan).ffill().dropna()
    print("  λ 최적화...")
    lam, best_r2 = optimize_lambda(Y, maturities)
    print(f"  최적 λ={lam:.4f}  R²={best_r2:.4f}  최대적재만기≈{1.7/lam:.1f}년")
    tau = np.asarray(maturities, dtype=float); Lam = _ns_loadings(tau, lam)
    beta_ols = np.array([np.linalg.lstsq(Lam, row.values.astype(float), rcond=None)[0]
                         for _, row in Y.iterrows()])
    X_v = beta_ols[:-1]; Z_v = beta_ols[1:]
    X_c = np.column_stack([np.ones(len(X_v)), X_v])
    coef = np.linalg.lstsq(X_c, Z_v, rcond=None)[0]
    c_vec = coef[0]; Phi = _make_stationary(coef[1:].T)
    Q_mat = np.cov((Z_v-X_c@coef).T) + np.eye(3)*1e-6
    H_mat = np.diag(np.maximum(np.nanvar(Y.values-beta_ols@Lam.T, axis=0), 1e-4))
    filt_states = np.zeros((len(Y), 3)); I3 = np.eye(3)
    a = beta_ols[0].copy(); P = np.eye(3)
    for t in range(len(Y)):
        y_t = Y.iloc[t].values.astype(float)
        if t > 0: a = c_vec+Phi@a; P = Phi@P@Phi.T+Q_mat
        S_mat = Lam@P@Lam.T+H_mat; K = P@Lam.T@np.linalg.pinv(S_mat)
        a = a+K@(y_t-Lam@a); P = (I3-K@Lam)@P
        filt_states[t] = a
    filt_factors = pd.DataFrame(filt_states, index=Y.index, columns=["L","S","C"])
    L_now, S_now, C_now = filt_states[-1]
    print(f"  현재 인자: L={L_now:.2f}  S={S_now:.2f}  C={C_now:.2f}")
    fc_states = []; a_h = filt_states[-1].copy()
    for _ in range(steps): a_h = c_vec+Phi@a_h; fc_states.append(a_h.copy())
    fc_df = pd.DataFrame(fc_states, columns=["L","S","C"])
    L_f, S_f, C_f = fc_df.iloc[-1]
    fc_yields = {float(m): float(_ns_yield_from_factors(m,L_f,S_f,C_f,lam)) for m in maturities}
    curr_yields = {float(m): float(_ns_yield_from_factors(m,L_now,S_now,C_now,lam)) for m in maturities}
    print(f"  예측(12M): L={L_f:.2f}  S={S_f:.2f}  C={C_f:.2f}")
    return fc_df, fc_yields, filt_factors, curr_yields, lam

def dns_to_factor_adjustments(filt_factors, fc_df, horizon_months, macro_df=None):
    if fc_df is None or filt_factors is None: return {}
    L_now = float(filt_factors["L"].iloc[-1]); S_now = float(filt_factors["S"].iloc[-1])
    h_idx = min(horizon_months-1, len(fc_df)-1)
    L_fc = float(fc_df["L"].iloc[h_idx]); S_fc = float(fc_df["S"].iloc[h_idx])
    d_level = L_fc-L_now; d_short = (L_fc+S_fc)-(L_now+S_now)
    inf_share = 0.40
    if macro_df is not None and "Inflation Factor" in macro_df.columns:
        try:
            inf_f = macro_df["Inflation Factor"].dropna()
            l_diff = filt_factors["L"].diff().dropna()
            common = l_diff.reindex(inf_f.index).dropna()
            inf_c = inf_f.reindex(common.index).dropna()
            l_c = common.reindex(inf_c.index).dropna()
            if len(l_c) > 24:
                corr = float(np.corrcoef(l_c.values, inf_c.values)[0,1])
                inf_share = float(np.clip(abs(corr), 0.30, 0.60))
        except Exception: pass
    d_level_inf = d_level * inf_share
    RAW_STD_BEI = 0.13; d_bei_proxy = 0.0
    if macro_df is not None and "Inflation Factor" in macro_df.columns:
        inf_factor = macro_df["Inflation Factor"].dropna()
        if len(inf_factor) > 0:
            d_bei_proxy = float(inf_factor.iloc[-1]) * RAW_STD_BEI
    d_real_approx = d_short - d_bei_proxy
    hist_L_sigma = float(filt_factors["L"].diff().std()) or 0.15
    hist_short_sigma = float((filt_factors["L"]+filt_factors["S"]).diff().std()) or 0.20
    inf_adj  = float(np.clip(d_level_inf/hist_L_sigma, -1.0, 1.0))
    real_adj = float(np.clip(d_real_approx/hist_short_sigma, -1.0, 1.0))
    print(f"\n  [DNS → 팩터 보정 v4.2]  H={horizon_months}M")
    print(f"    ΔLevel={d_level:+.3f}%  inf_share={inf_share:.0%}  → inf_adj={inf_adj:+.3f}σ")
    print(f"    Δ(L+S)={d_short:+.3f}%  ΔBEI_proxy={d_bei_proxy:+.3f}%  → real_adj={real_adj:+.3f}σ")
    print(f"    ※ DNS = 기간구조 통계적 평균회귀 보조 신호 (시장 forward rate 아님)")
    return {"Inflation Factor": inf_adj, "Real Rate Factor": real_adj}

def apply_dns_to_scenarios(scenarios, dns_factor_adj):
    if not dns_factor_adj: return scenarios
    adjusted = {}; print("\n  [DNS 시나리오 보정]")
    for sc_name, sc in scenarios.items():
        new_deltas = dict(sc["deltas"])
        for factor, adj in dns_factor_adj.items():
            if factor in new_deltas:
                old_val = new_deltas[factor]
                new_val = float(np.clip(old_val+adj*0.5, -3.0, 3.0))
                new_deltas[factor] = new_val
                print(f"    [{sc_name}] {factor}: {old_val:+.2f}+{adj*0.5:+.2f}={new_val:+.2f}σ")
        adjusted[sc_name] = {**sc, "deltas": new_deltas}
    return adjusted


# ============================================================
# ■ STEP 6: 시나리오 Alpha 계산
# ============================================================
def compute_scenario_alpha_annual(regression_results, scenarios):
    """
    시나리오 alpha 산출 — 6개월 기준 (v4.3-6M).

    3M demeaned 회귀 계수 × 2 = 6개월 alpha
      (3M × 2 = 6M, HORIZON_MONTHS=6 기준)

    반환값 단위: 6개월 소수점 수익률
    """
    ann_mult = HORIZON_MONTHS // 3   # 6 // 3 = 2  (3M 회귀 → 6M)
    print(f"\n[Step 6] 시나리오 Alpha ({HORIZON_MONTHS}M 기준, ×{ann_mult})...")
    all_alpha = {}
    for sc_name, sc in scenarios.items():
        alpha_dict = {}
        for asset, reg in regression_results.items():
            betas = reg["betas"]; raw_alpha = 0.0
            for factor_name, delta in sc["deltas"].items():
                matched = [c for c in betas.index if factor_name in c and c != "const"]
                if matched:
                    bc = max(matched, key=lambda c: abs(betas[c]))
                    raw_alpha += betas[bc] * delta
            alpha_dict[asset] = (raw_alpha * ann_mult) / 100.0
        all_alpha[sc_name] = pd.Series(alpha_dict)
        print(f"  [{sc_name}]  ({HORIZON_MONTHS}M 기준)")
        for a, v in all_alpha[sc_name].items():
            v_ann = (1 + v) ** (12 / HORIZON_MONTHS) - 1
            print(f"    {a:>5s}: {v*100:+.2f}% ({HORIZON_MONTHS}M)  "
                  f"≈ {v_ann*100:+.2f}% (연환산)")
    return all_alpha

def compute_scenario_returns_for_viz(regression_results, scenarios):
    """시각화용 H=3M, H=6M 기대수익률 (6M 기준 스케일)"""
    all_results = {}
    for H in SCENARIO_HORIZONS:
        sc_rets = {}
        for sc_name, sc in scenarios.items():
            asset_rets = {}
            for asset, reg in regression_results.items():
                betas = reg["betas"]; raw_alpha = 0.0
                for factor_name, delta in sc["deltas"].items():
                    matched = [c for c in betas.index if factor_name in c and c != "const"]
                    if matched:
                        bc = max(matched, key=lambda c: abs(betas[c]))
                        raw_alpha += betas[bc] * delta
                # 3M 회귀 → H개월: ÷3 × H
                monthly_equiv = raw_alpha / 3
                r_H = (1.0 + monthly_equiv / 100.0) ** H - 1.0
                asset_rets[asset] = np.clip(r_H, -1.0, 1.0)
            sc_rets[sc_name] = pd.Series(asset_rets)
        all_results[H] = sc_rets
    return all_results


# ============================================================
# ■ STEP 7: Prior = R² 기반 블렌딩 (역사적 수익률 + 팩터 기대수익률)
# ============================================================
def compute_blended_prior(monthly_rets_1m, scenario_alpha_annual,
                          sc_name, assets, regression_results,
                          lookback=None):
    """
    Prior(Π) — 6개월 기준 (v4.3-6M).

    Prior_6m[i] = mu_hist_6m[i] + R²[i] × alpha_6m[i]

      mu_hist_6m = 월간 평균수익률 × 6   (6M 기준)
      alpha_6m   = 3M 회귀 alpha × 2    (6M 기준, 입력값과 동일 단위)
      R²[i]      = 팩터 회귀 결정계수   (0~1)
    """
    rets = monthly_rets_1m[assets].dropna(how="all") / 100.0
    if lookback: rets = rets.iloc[-lookback:]
    mu_hist_6m = rets.mean() * HORIZON_MONTHS      # 월간 × 6
    alpha_sc   = scenario_alpha_annual.get(sc_name, pd.Series(dtype=float))
    blended = {}
    r2_used = {}
    sc_label = f"{sc_name} [{HORIZON_MONTHS}M 기준]"
    date_range = f"{rets.index[0].date()}~{rets.index[-1].date()}"
    print(f"  [Prior 블렌딩 - {sc_label}  {date_range}]")
    print("  자산   mu_hist_6m   alpha_6m   R2(w)   Prior_6m   Prior_ann")
    for a in assets:
        mu_a    = float(mu_hist_6m.get(a, 0))
        alpha_a = float(alpha_sc.get(a, 0))   # 6M 기준
        r2 = float(np.clip(
                regression_results.get(a, {}).get("r2", 0), 0.0, 1.0)
             ) if USE_FACTOR_RETURN else 0.0
        pi_a    = mu_a + r2 * alpha_a
        blended[a] = pi_a
        r2_used[a] = r2
        pi_ann  = (1 + pi_a) ** (12 / HORIZON_MONTHS) - 1
        print(f"  {a:>5s}  {mu_a*100:>10.2f}%  {alpha_a*100:>8.2f}%  "
              f"{r2:>6.3f}   {pi_a*100:>8.2f}%  {pi_ann*100:>9.2f}%")
    return pd.Series(blended), r2_used



# ============================================================
# ■ STEP 8: 공분산 (v4.3 신규: 자산군별 블렌딩)
# ============================================================
def build_ewma_covariance(prices, assets):
    """
    역사적 EWMA 공분산 — 6개월 기준으로 변환 (v4.3-6M).

    연환산 공분산 × (HORIZON_MONTHS/12) = 6M 기준 공분산
    """
    port_prices = prices[assets].dropna()
    if USE_EWMA_COV:
        S_ann = risk_models.exp_cov(port_prices, span=EWMA_SPAN, frequency=252)
        print(f"  EWMA 공분산 (span={EWMA_SPAN}d) → 6M 스케일 ×{HORIZON_MONTHS/12:.1f}")
    else:
        S_ann = risk_models.CovarianceShrinkage(port_prices).ledoit_wolf()
        print(f"  Ledoit-Wolf 공분산 → 6M 스케일 ×{HORIZON_MONTHS/12:.1f}")
    # 연환산 → 6M 기준 변환
    scale = HORIZON_MONTHS / 12.0   # = 0.5
    S_6m_vals = S_ann.values * scale
    S_6m = pd.DataFrame(S_6m_vals, index=S_ann.index, columns=S_ann.columns)
    return S_6m

def build_factor_covariance(regression_results, macro_df, assets):
    """
    팩터 β 기반 공분산 (Barra 스타일).

    Σ_factor = B × Σ_F × B' + D

    B    : (N × K) 베타 행렬 — 회귀에서 직접 추출
    Σ_F  : (K × K) 팩터 공분산 — macro_df에서 추정
    D    : (N × N) 잔차 분산 대각행렬

    단위 통일:
      베타는 3M 수익률(%) 기준 → 연환산 ×4, % → 소수점 /100²
    """
    factor_cols = []
    for f, lag in FACTOR_LAGS.items():
        if f in macro_df.columns:
            factor_cols.append(f"{f}_L{lag}")

    # 3M 회귀 → HORIZON_MONTHS 기준 스케일 (=2)
    ann_mult = HORIZON_MONTHS // 3

    # ── B 행렬 & D 대각행렬 구성 ────────────────────────────
    B_rows = []; D_diag = []
    for asset in assets:
        if asset not in regression_results:
            B_rows.append(np.zeros(len(factor_cols)))
            D_diag.append(0.04)  # fallback: σ ≈ 20%
            continue
        reg   = regression_results[asset]
        betas = reg["betas"]
        row   = np.array([float(betas.get(fc, 0.0)) for fc in factor_cols])
        B_rows.append(row)
        # 잔차 분산 연환산 → 소수점²
        resid_var = float(reg["resid"].var()) * ann_mult / (100.0**2)
        D_diag.append(max(resid_var, 1e-6))

    B = np.array(B_rows)   # (N × K)
    D = np.diag(D_diag)    # (N × N)

    # ── Σ_F: 팩터 수익률 공분산 ────────────────────────────
    F_data = pd.DataFrame(index=macro_df.dropna().index)
    for fc in factor_cols:
        base = fc.rsplit("_L", 1)[0]
        lag  = int(fc.rsplit("_L", 1)[1])
        if base in macro_df.columns:
            F_data[fc] = macro_df[base].shift(lag)
    F_data = F_data.dropna()
    Sigma_F = F_data.cov().values if len(F_data) > 10 else np.eye(len(factor_cols))

    # ── Σ_factor = B × Σ_F × B' + D ──────────────────────
    # B: % / σ 단위 → HORIZON_MONTHS 기준 ×ann_mult / 100² 변환
    scale = ann_mult / (100.0 ** 2)
    Sigma_factor = scale * (B @ Sigma_F @ B.T) + D
    Sigma_factor = _ensure_psd(Sigma_factor)

    Sigma_factor_df = pd.DataFrame(Sigma_factor, index=assets, columns=assets)

    vols = pd.Series(np.sqrt(np.diag(Sigma_factor)), index=assets)
    print(f"\n  [팩터 Σ_factor]  B({len(assets)}×{len(factor_cols)})")
    for a in assets:
        atype = ASSETS[a]["type"]
        print(f"    {a:>5s}: σ_factor={vols[a]*100:.2f}%  ({atype})")

    return Sigma_factor_df

def build_blended_covariance(S_ewma, S_factor, assets, regression_results):
    """
    R² 기반 자동 공분산 블렌딩 (v4.4).

    W_ij = (R²[i] + R²[j]) / 2      ← 두 자산 R² 산술평균
    Σ_blend[i,j] = W_ij × Σ_factor[i,j] + (1-W_ij) × Σ_EWMA[i,j]

    R²[i]: 회귀 결정계수 (adj-R² 사용, 0~1 클램프)
      - 두 자산 모두 R² 높아야 Σ_factor 비중 높음
      - 한 자산이라도 R² 낮으면 보수적으로 Σ_EWMA 비중 높아짐
    """
    r2_arr = np.array([
        float(np.clip(regression_results.get(a, {}).get("r2", 0), 0.0, 1.0))
        for a in assets
    ])

    # W_ij 행렬
    W_factor = (r2_arr[:, None] + r2_arr[None, :]) / 2.0   # (N×N)
    W_hist   = 1.0 - W_factor

    S_ewma_vals   = S_ewma.values
    S_factor_vals = S_factor.values

    S_blend_vals = W_factor * S_factor_vals + W_hist * S_ewma_vals
    S_blend_vals = _ensure_psd(S_blend_vals)
    S_blend = pd.DataFrame(S_blend_vals, index=assets, columns=assets)

    vol_ewma   = np.sqrt(np.diag(S_ewma_vals))
    vol_factor = np.sqrt(np.diag(S_factor_vals))
    vol_blend  = np.sqrt(np.diag(S_blend_vals))

    print(f"\n  [공분산 블렌딩 v4.4]  R² 자동 가중")
    print(f"  {'자산':>5s}  {'R²(w)':>6s}  {'σ_EWMA':>8s}  "
          f"{'σ_factor':>9s}  {'σ_blend':>8s}")
    for i, a in enumerate(assets):
        print(f"  {a:>5s}  {r2_arr[i]:>5.3f}   "
              f"{vol_ewma[i]*100:>7.2f}%  "
              f"{vol_factor[i]*100:>8.2f}%  "
              f"{vol_blend[i]*100:>7.2f}%")
    return S_blend

def build_covariance(prices, assets, regression_results=None, macro_df=None):
    """
    공분산 행렬 최종 산출 (v4.4).

    USE_FACTOR_COV=True:
      Σ_blend[i,j] = W_ij × Σ_factor[i,j] + (1-W_ij) × Σ_EWMA[i,j]
      W_ij = (R²[i] + R²[j]) / 2  (R² 자동 가중)
    USE_FACTOR_COV=False:
      순수 EWMA
    """
    S_ewma = build_ewma_covariance(prices, assets)

    if USE_FACTOR_COV and regression_results is not None and macro_df is not None:
        S_factor = build_factor_covariance(regression_results, macro_df, assets)
        S        = build_blended_covariance(S_ewma, S_factor, assets,
                                            regression_results)
    else:
        if USE_FACTOR_COV:
            print("  ⚠ USE_FACTOR_COV=True이지만 regression_results/macro_df 없음 → EWMA만 사용")
        S = S_ewma

    vols = pd.Series(np.sqrt(np.diag(S.values)), index=S.index)
    print(f"\n  [최종 Σ 변동성 요약]")
    for a in assets:
        if a in vols.index:
            bf = " ←채권" if ASSETS[a]["type"].startswith("bond") else ""
            print(f"  {a:>5s}: σ={vols[a]*100:.2f}%{bf}")
    return S


# ============================================================
# ■ STEP 8: BL Views + 최적화
# ============================================================
def build_views_for_scenario(sc_name, assets, S, regression_results):
    """
    Views(Q) = USER_BL_VIEWS 직접 입력값만.

    - USER_BL_VIEWS에 등록된 자산만 Q·P·Ω에 포함
    - 등록되지 않은 자산은 Prior(Π)로만 결정 (BL이 Π를 그대로 유지)
    - P 행렬: 단위행렬 대신 사용자 뷰 있는 자산만 선택하는 행렬
    - USER_BL_VIEWS가 비어 있으면 (Q, Omega, P) = None → BL Prior만 사용
    """
    sc_views = USER_BL_VIEWS.get(sc_name, {})

    if not sc_views:
        print(f"    [{sc_name}] USER_BL_VIEWS 없음 → Prior만 사용")
        return None, None, None

    n = len(assets)
    view_assets = [a for a in assets if a in sc_views]
    k = len(view_assets)

    # P: (k × n) 선택 행렬 — 뷰 있는 자산 행만
    P = np.zeros((k, n))
    for row_i, a in enumerate(view_assets):
        col_j = assets.index(a)
        P[row_i, col_j] = 1.0

    # Q: 직접 입력 연간 수익률
    Q = np.array([sc_views[a]["annual_return"] for a in view_assets])

    # Omega: confidence 기반 대각행렬
    S_diag = np.diag(S.values)
    diag = []
    for a in view_assets:
        j    = assets.index(a)
        conf = np.clip(sc_views[a]["confidence"], 1e-6, 1-1e-6)
        diag.append(TAU * S_diag[j] * (1-conf)/conf)
    Omega = np.diag(diag)

    print(f"    [{sc_name}] USER_BL_VIEWS: {view_assets}")
    for a in view_assets:
        print(f"      {a:>5s}: Q={sc_views[a]['annual_return']*100:.2f}%  "
              f"conf={sc_views[a]['confidence']:.0%}")

    return Q, Omega, P

def compute_mdd(prices, assets, lookback=None):
    try:   mo = prices.resample("ME").last()
    except: mo = prices.resample("M").last()
    if lookback: mo = mo.iloc[-lookback:]
    mdd_dict = {}; lb = f"최근 {lookback}M" if lookback else "전체"
    print(f"\n  [MDD] {mo.index[0].date()}~{mo.index[-1].date()} ({lb})")
    for a in assets:
        if a not in mo.columns: mdd_dict[a] = -0.20; continue
        p = mo[a].dropna(); cum_max = p.expanding().max()
        dd = (p-cum_max)/cum_max; mdd_dict[a] = float(dd.min())
        print(f"  {a:>5s}: MDD={mdd_dict[a]*100:.1f}%")
    return pd.Series(mdd_dict)

def compute_mdd_wmax(mdd, assets):
    mdd_abs = mdd.abs(); median_mdd = float(mdd_abs.median())
    wmax_dict = {}
    print(f"\n  [MDD→w_max] BASE={MDD_BASE_MAX:.2f}  중앙={median_mdd*100:.1f}%")
    for a in assets:
        raw  = MDD_BASE_MAX * (median_mdd / max(mdd_abs[a], 1e-6))
        wmax = float(np.clip(raw, MDD_WMAX_FLOOR, MDD_WMAX_CAP))
        wmax = max(wmax, MIN_WEIGHT_ALL+0.001)
        wmax_dict[a] = wmax
        print(f"  {a:>5s}: MDD={mdd[a]*100:.1f}%  w_max={wmax*100:.1f}%")
    return pd.Series(wmax_dict)

def _build_weight_bounds(prices, assets, w_max_global=0.40):
    n = len(assets)
    if USE_MDD_WMAX:
        mdd    = compute_mdd(prices, assets, MDD_LOOKBACK)
        wmax_s = compute_mdd_wmax(mdd, assets)
        wmax_v = [float(wmax_s.get(a, w_max_global)) for a in assets]
    else:
        wmax_v = [w_max_global] * n
    bond_idx = [i for i,a in enumerate(assets) if ASSETS[a]["type"].startswith("bond")]
    n_bonds  = len(bond_idx)
    bond_lo  = (BOND_TOTAL_MIN/n_bonds) if n_bonds > 0 else MIN_WEIGHT_ALL
    weight_bounds = []
    for i, a in enumerate(assets):
        lo = max(bond_lo, MIN_WEIGHT_ALL) if i in bond_idx else MIN_WEIGHT_ALL
        hi = max(wmax_v[i], lo+0.001)
        weight_bounds.append((lo, hi))
    print(f"    weight_bounds: 채권lo={bond_lo*100:.1f}%  비채권lo={MIN_WEIGHT_ALL*100:.0f}%")
    return weight_bounds

def _optimize_single(mu_bl, S_bl, prices, w_max_global=0.40):
    assets = PORTFOLIO_ASSETS
    weight_bounds = _build_weight_bounds(prices, assets, w_max_global)
    for sharpe_mode in [True, False]:
        try:
            ef = EfficientFrontier(mu_bl, S_bl, weight_bounds=weight_bounds)
            if sharpe_mode: ef.max_sharpe(risk_free_rate=RISK_FREE_RATE_6M)
            else:           ef.min_volatility()
            cleaned = ef.clean_weights()
            perf    = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE_6M)
            bond_w  = sum(cleaned.get(a,0) for a in assets if ASSETS[a]["type"].startswith("bond"))
            min_w   = min(cleaned.values())
            mode_s  = "Max Sharpe" if sharpe_mode else "Min Vol (fallback)"
            print(f"    [{mode_s}]  채권={bond_w*100:.1f}%  최소비중={min_w*100:.1f}%")
            return cleaned, perf
        except Exception as e:
            print(f"    ⚠ {'Max Sharpe' if sharpe_mode else 'Min Vol'} 실패({e})")
    raise RuntimeError("최적화 실패")


# ============================================================
# ■ STEP 9: 확률 가중 블렌딩 포트폴리오
# ============================================================
def compute_blended_portfolio(sc_results, scenario_probs, prices):
    total_p = sum(scenario_probs.values())
    probs   = {k: v/total_p for k,v in scenario_probs.items()}
    print(f"\n[Step 9] 확률 가중 블렌딩 포트폴리오")
    for sc_name, p in probs.items(): print(f"  {sc_name}: {p*100:.1f}%")
    assets = PORTFOLIO_ASSETS; n = len(assets)
    mu_blend_vals = np.zeros(n); S_blend_vals = np.zeros((n,n))
    for sc_name, p in probs.items():
        if sc_name not in sc_results: continue
        mu_arr = np.array([sc_results[sc_name]["mu_bl"].get(a,0) for a in assets])
        mu_blend_vals += p * mu_arr
        S_blend_vals  += p * sc_results[sc_name]["S_bl"].values
    mu_blend = pd.Series(mu_blend_vals, index=assets)
    S_blend  = pd.DataFrame(S_blend_vals, index=assets, columns=assets)
    print(f"\n  μ_blend:")
    for a in assets:
        on_mu  = sc_results["Risk-On"]["mu_bl"].get(a,0)  if "Risk-On"  in sc_results else 0
        off_mu = sc_results["Risk-Off"]["mu_bl"].get(a,0) if "Risk-Off" in sc_results else 0
        print(f"    {a:>5s}: {mu_blend[a]*100:+.2f}%  (On={on_mu*100:+.2f}%  Off={off_mu*100:+.2f}%)")
    weight_bounds = _build_weight_bounds(prices, assets)
    cleaned = perf = None
    for sharpe_mode in [True, False]:
        try:
            ef = EfficientFrontier(mu_blend, S_blend, weight_bounds=weight_bounds)
            if sharpe_mode: ef.max_sharpe(risk_free_rate=RISK_FREE_RATE_6M)
            else:           ef.min_volatility()
            cleaned = ef.clean_weights()
            perf    = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE_6M)
            bond_w  = sum(cleaned.get(a,0) for a in assets if ASSETS[a]["type"].startswith("bond"))
            mode_s  = "Max Sharpe" if sharpe_mode else "Min Vol (fallback)"
            print(f"\n  [{mode_s}]  μ={perf[0]*100:.2f}%  σ={perf[1]*100:.2f}%  Sharpe={perf[2]:.3f}  채권={bond_w*100:.1f}%")
            break
        except Exception as e:
            print(f"  ⚠ {'Max Sharpe' if sharpe_mode else 'Min Vol'} 실패({e})")
    if cleaned is None: raise RuntimeError("블렌딩 최적화 실패")
    return cleaned, perf, mu_blend, S_blend


# ============================================================
# ■ STEP 8 메인: 시나리오별 BL + 최적화
# ============================================================
def run_all_scenario_optimizations(prices, regression_results,
                                   scenario_alpha_annual,
                                   scenarios, port_rets_1m,
                                   dns_fc_df=None, dns_filt_factors=None,
                                   macro_df=None):
    """
    [v4.3 수정]
    Prior(Π) = R²[i] × mu_factor[i] + (1-R²[i]) × mu_hist[i]
             = mu_hist[i] + R²[i] × scenario_alpha[i]
    Views(Q) = USER_BL_VIEWS 직접 입력값만
               USER_BL_VIEWS 없으면 BL이 Prior를 그대로 사용
    """
    print("\n[Step 8] 시나리오별 BL + 최적화 (v4.3)...")
    assets = PORTFOLIO_ASSETS

    # ── 공분산: 자산군별 블렌딩 ──────────────────────────────
    S = build_covariance(prices, assets,
                         regression_results=regression_results,
                         macro_df=macro_df)

    # ── DNS 팩터 보정 ─────────────────────────────────────────
    dns_factor_adj = dns_to_factor_adjustments(
        dns_filt_factors, dns_fc_df, 6, macro_df=macro_df)
    dns_scenarios  = apply_dns_to_scenarios(scenarios, dns_factor_adj)

    if dns_factor_adj:
        print("\n  DNS 보정 후 alpha 재계산...")
        scenario_alpha_dns = compute_scenario_alpha_annual(
            regression_results, dns_scenarios)
    else:
        scenario_alpha_dns = scenario_alpha_annual

    sc_results = {}
    for sc_name, sc in dns_scenarios.items():
        print(f"\n  ── [{sc_name}]  p={sc['prob']*100:.0f}% ───────────────────")

        # ── Prior: R² 기반 역사적 + 팩터 alpha 블렌딩 ──────
        Pi, r2_used = compute_blended_prior(
            port_rets_1m, scenario_alpha_dns,
            sc_name, assets, regression_results, HIST_LOOKBACK)

        # ── Views: USER_BL_VIEWS 직접입력만 ─────────────────
        Q, Omega, P = build_views_for_scenario(
            sc_name, assets, S, regression_results)

        # ── BL 모형 실행 ─────────────────────────────────────
        if Q is None:
            # 직접 입력 View 없음 → Prior가 곧 기대수익률
            mu_bl = Pi
            bl    = None
            S_bl  = S
            print(f"    Views 없음 → μ_BL = Prior 그대로 사용")
        else:
            bl    = BlackLittermanModel(
                        cov_matrix=S, pi=Pi,
                        P=P, Q=Q, omega=Omega, tau=TAU)
            mu_bl = bl.bl_returns()
            S_bl  = bl.bl_cov()

        # ── 비교 출력 ─────────────────────────────────────────
        print(f"    {'자산':>5s}  {'Prior(Π)':>9s}  {'μ_BL':>9s}  {'Δ':>8s}")
        for a in assets:
            pi_a  = float(Pi.get(a, 0))
            mbl_a = float(mu_bl.get(a, 0)) if hasattr(mu_bl,'get') else float(mu_bl[a])
            print(f"    {a:>5s}  {pi_a*100:>8.2f}%  {mbl_a*100:>8.2f}%  "
                  f"{(mbl_a-pi_a)*100:>+7.2f}%")

        cleaned, perf = _optimize_single(mu_bl, S_bl, prices)
        r_ann, s_ann, sh_ann = _annualize_perf_6m(perf)
        print(f"\n    μ_BL={perf[0]*100:.2f}% (6M)  → 연환산 {r_ann*100:.2f}%  σ_ann={s_ann*100:.2f}%  Sharpe={sh_ann:.3f}")
        print("    비중: " + "  ".join(
            f"{a}:{v*100:.1f}%" for a,v in cleaned.items() if v>0.005))

        latest = get_latest_prices(prices[PORTFOLIO_ASSETS])
        da = DiscreteAllocation(cleaned, latest,
                                total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
        alloc, leftover = da.greedy_portfolio()

        sc_results[sc_name] = {
            "Pi": Pi, "Q": Q, "r2_used": r2_used,
            "mu_bl": mu_bl, "S_bl": S_bl, "bl": bl,
            "weights": cleaned, "perf": perf,
            "alloc": alloc, "leftover": leftover,
        }
    return sc_results, sc_results[list(sc_results.keys())[0]]["Pi"], S


# ============================================================
# ■ STEP 10: VISUALIZATION
# ============================================================
def _annualize_perf_6m(perf_6m):
    """
    6개월 기준 최적화 성과 → 연환산 변환.

    r_6m, sigma_6m, sharpe_6m (6M 기준) → r_ann, sigma_ann, sharpe_ann

    r_ann   = (1 + r_6m)^(12/6) - 1   = (1 + r_6m)² - 1
    σ_ann   = σ_6m × √(12/6)           = σ_6m × √2
    Sharpe  = (r_ann - RISK_FREE_RATE) / σ_ann
    """
    r_6m, sigma_6m = perf_6m[0], perf_6m[1]
    scale = 12 / HORIZON_MONTHS          # = 2
    r_ann    = (1 + r_6m) ** scale - 1
    sigma_ann = sigma_6m * np.sqrt(scale)
    sharpe_ann = (r_ann - RISK_FREE_RATE) / sigma_ann if sigma_ann > 0 else 0.0
    return r_ann, sigma_ann, sharpe_ann


def _savefig(fig, fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"  ✓ 저장 → {fname}")

def plot_return_decomposition(monthly_rets_1m, scenario_alpha_annual,
                              sc_results, regression_results):
    """
    [Fig 0] 기대수익률 구성 요소 분해 시각화 (v4.4 R² 자동 가중).

    행 1 — 기대수익률 분해:
      ▸ mu_hist          : 역사적 평균수익률 (회색 베이스 막대)
      ▸ R² × alpha       : R² 가중 alpha 기여분 (시나리오색, stacked)
      ▸ Prior (Π)        : mu_hist + R²×alpha (★ 마커)
      ▸ USER_BL_VIEWS    : 직접 입력값 (◆ 마커, 있을 때만)
      ▸ SE(mu_hist)      : 역사적 평균 표준오차 (에러바)

    행 2 — 불확실성(표준편차) 분해:
      ▸ σ_hist    : 역사적 연환산 변동성
      ▸ σ_alpha   : β 추정 SE × delta 합산 → alpha 추정 불확실성
      ▸ σ_prior   : √(σ_hist² × (1-R²)² + σ_factor² × R²²) 근사
                    = R² 가중 블렌딩된 Prior의 표준편차
    """
    print("\n[Fig 0] 기대수익률 분해 시각화 (R² 자동 가중)...")
    assets   = PORTFOLIO_ASSETS
    n        = len(assets)
    x        = np.arange(n)
    ann_mult = HORIZON_MONTHS // 3   # 3M → 6M: ×2

    # ── 역사적 수익률 ───────────────────────────────────────
    rets     = monthly_rets_1m[assets].dropna(how="all") / 100.0
    rets     = rets.iloc[-HIST_LOOKBACK:] if HIST_LOOKBACK else rets
    mu_hist  = rets.mean() * 12
    se_hist  = rets.sem()  * 12
    std_hist = rets.std()  * np.sqrt(12)

    # ── 자산별 R² ───────────────────────────────────────────
    r2_arr = np.array([
        float(np.clip(regression_results.get(a, {}).get("r2", 0), 0.0, 1.0))
        for a in assets
    ])

    # ── alpha 표준편차 (β SE × delta 합산) ──────────────────
    def _alpha_std(sc_name):
        sc = SCENARIOS[sc_name]
        std_dict = {}
        for a in assets:
            if a not in regression_results:
                std_dict[a] = 0.0; continue
            reg = regression_results[a]
            model = reg.get("model")
            var_sum = 0.0
            for factor_name, delta in sc["deltas"].items():
                matched = [c for c in reg["betas"].index
                           if factor_name in c and c != "const"]
                if not matched: continue
                bc = max(matched, key=lambda c: abs(reg["betas"][c]))
                try:
                    se_b = float(model.bse[bc]) if model is not None else 0.0
                except Exception:
                    se_b = 0.0
                var_sum += (se_b * delta) ** 2
            std_dict[a] = np.sqrt(var_sum) * ann_mult / 100.0
        return pd.Series(std_dict)

    sc_names = list(SCENARIOS.keys())
    colors   = {"Risk-On": "#2ecc71", "Risk-Off": "#e74c3c"}
    alpha_c  = {"Risk-On": "#27ae60", "Risk-Off": "#c0392b"}

    fig, axes = plt.subplots(2, 2, figsize=(22, 14),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})

    for col, sc_name in enumerate(sc_names):
        sc_color = colors[sc_name]
        al_color = alpha_c[sc_name]

        alpha_sc   = scenario_alpha_annual.get(sc_name, pd.Series(dtype=float))
        mu_hist_v  = np.array([float(mu_hist.get(a, 0))   for a in assets])
        alpha_v    = np.array([float(alpha_sc.get(a, 0))  for a in assets])
        r2_alpha_v = r2_arr * alpha_v        # R² × alpha 기여분
        prior_v    = mu_hist_v + r2_alpha_v  # Prior(Π)

        user_views = USER_BL_VIEWS.get(sc_name, {})
        user_vals  = [user_views[a]["annual_return"] if a in user_views
                      else np.nan for a in assets]

        # ── 행 1: 기대수익률 ─────────────────────────────────
        ax = axes[0, col]
        ax.bar(x, mu_hist_v * 100, width=0.55,
               color="#bdc3c7", alpha=0.85, edgecolor="white",
               label="역사적 평균수익률 (mu_hist)")

        pos_mask = r2_alpha_v >= 0; neg_mask = ~pos_mask
        if pos_mask.any():
            ax.bar(x[pos_mask], r2_alpha_v[pos_mask] * 100,
                   bottom=mu_hist_v[pos_mask] * 100,
                   width=0.55, color=al_color, alpha=0.75, edgecolor="white",
                   label="R²×alpha 기여 (양수)")
        if neg_mask.any():
            ax.bar(x[neg_mask], r2_alpha_v[neg_mask] * 100,
                   bottom=mu_hist_v[neg_mask] * 100,
                   width=0.55, color="#e67e22", alpha=0.75, edgecolor="white",
                   label="R²×alpha 기여 (음수)")

        ax.scatter(x, prior_v * 100, marker="*", s=220,
                   color=sc_color, zorder=6, edgecolors="black",
                   linewidths=0.5, label="Prior(Π) = mu_hist + R²×alpha")

        uv_arr = np.array(user_vals, dtype=float)
        valid  = ~np.isnan(uv_arr)
        if valid.any():
            ax.scatter(x[valid], uv_arr[valid] * 100,
                       marker="D", s=100, color="#9b59b6", zorder=7,
                       edgecolors="black", linewidths=0.6,
                       label="USER_BL_VIEWS (직접 입력)")

        ax.errorbar(x - 0.10, mu_hist_v * 100,
                    yerr=se_hist.values * 100,
                    fmt="none", ecolor="#7f8c8d",
                    elinewidth=1.2, capsize=3, alpha=0.7,
                    label="SE (역사적 평균)")

        # R² 레이블 (막대 아래)
        for i, r2 in enumerate(r2_arr):
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] < -1 else
                    min(mu_hist_v[i], prior_v[i]) * 100 - 1.5,
                    f"R²={r2:.2f}", ha="center", fontsize=6.5,
                    color="#666666")

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(assets, fontsize=9)
        ax.set_ylabel("연간 기대수익률 (%)")
        ax.set_title(f"[{sc_name}]  기대수익률 구성 분해  (R² 자동 가중)\n"
                     f"p={SCENARIO_PROBS[sc_name]*100:.0f}%  "
                     f"— {SCENARIOS[sc_name]['description'][:30]}",
                     fontsize=10, fontweight="bold", color=sc_color)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.20, axis="y")

        for i, pv in enumerate(prior_v):
            ax.text(i, pv * 100 + (0.4 if pv >= 0 else -1.2),
                    f"{pv*100:+.1f}%", ha="center", fontsize=7,
                    color=sc_color, fontweight="bold")

        # ── 행 2: 표준편차 ───────────────────────────────────
        ax2 = axes[1, col]
        std_a = _alpha_std(sc_name)

        std_hist_v  = std_hist.values * 100
        std_alpha_v = np.array([float(std_a.get(a, 0)) for a in assets]) * 100
        # Prior σ: R² 가중 블렌딩
        # σ_prior ≈ sqrt((1-R²)²×σ_hist² + R²²×σ_alpha²)
        std_prior_v = np.sqrt(
            ((1 - r2_arr) * std_hist_v) ** 2 +
            (r2_arr       * std_alpha_v) ** 2
        )

        bw = 0.25
        ax2.bar(x - bw, std_hist_v,  bw, label="σ_hist (역사적 변동성)",
                color="#3498db", alpha=0.80, edgecolor="white")
        ax2.bar(x,      std_alpha_v, bw, label="σ_alpha (β 추정 불확실성)",
                color=al_color,    alpha=0.80, edgecolor="white")
        ax2.bar(x + bw, std_prior_v, bw, label="σ_prior (R² 가중 합산)",
                color=sc_color,    alpha=0.80, edgecolor="white")

        # R² 레이블
        for i, r2 in enumerate(r2_arr):
            ax2.text(i, max(std_hist_v[i], std_alpha_v[i], std_prior_v[i]) + 0.3,
                     f"R²={r2:.2f}", ha="center", fontsize=6.5, color="#666666")

        ax2.set_xticks(x); ax2.set_xticklabels(assets, fontsize=9)
        ax2.set_ylabel("연간 표준편차 (%)")
        ax2.set_title(f"[{sc_name}]  불확실성(표준편차) 분해  (R² 자동 가중)",
                      fontsize=10, fontweight="bold", color=sc_color)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.20, axis="y")

    fig.suptitle(
        "기대수익률 구성 분해  v4.4  |  R² 자동 가중\n"
        "■ 회색=mu_hist  ■ 색=R²×alpha 기여  ★=Prior(Π)  ◆=직접입력\n"
        "하단: σ_hist / σ_alpha(β추정오차) / σ_prior(R² 가중 합산)  "
        "— 각 막대 위 R²값 표시",
        fontsize=12, fontweight="bold")
    _savefig(fig, "fig0_return_decomposition.png")


def plot_macro_analysis(macro_df, regression_results):
    print("\n[Fig 1] 매크로 분석...")
    assets = PORTFOLIO_ASSETS
    fig = plt.figure(figsize=(22,15))
    gs  = gridspec.GridSpec(2,2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0,:])
    fc_colors = {"Inflation Factor":"#e74c3c","Real Rate Factor":"#3498db",
                 "Credit Factor":"#f39c12","Growth Factor":"#2ecc71",
                 "Labor Factor":"#9b59b6","Commodity Factor":"#e67e22"}
    for col in macro_df.columns:
        ax1.plot(macro_df.index, macro_df[col], label=col, lw=1.4, alpha=0.85,
                 color=fc_colors.get(col,"gray"))
    ax1.axhline(0,color="k",lw=0.8,ls="--",alpha=0.4)
    ax1.fill_between(macro_df.index,-1,1,alpha=0.04,color="gray")
    ax1.set_title("매크로 6팩터 시계열 (Z-score)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, ncol=3); ax1.grid(True, alpha=0.25)
    ax2 = fig.add_subplot(gs[1,0])
    r2v = {a: regression_results[a]["adj_r2"] for a in assets if a in regression_results}
    srt = dict(sorted(r2v.items(), key=lambda x: x[1]))
    col2 = ["#2ecc71" if v>0.20 else ("#f39c12" if v>0.10 else "#e74c3c") for v in srt.values()]
    bars = ax2.barh(list(srt.keys()), list(srt.values()), color=col2, edgecolor="white")
    ax2.axvline(0.20, color="red", ls="--", lw=1.2, label="기준 0.20")
    ax2.set_title("팩터 설명력 (adj-R²)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25, axis="x")
    for bar, val in zip(bars, srt.values()):
        ax2.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    ax3 = fig.add_subplot(gs[1,1])
    beta_data = {}
    for asset in assets:
        if asset not in regression_results: continue
        betas, cb = regression_results[asset]["betas"], {}
        for col, val in betas.items():
            if col=="const": continue
            base = col.rsplit("_L",1)[0]
            if base not in cb or abs(val)>abs(cb[base]): cb[base] = val
        beta_data[asset] = cb
    beta_df = pd.DataFrame(beta_data).T.fillna(0)
    sns.heatmap(beta_df, annot=True, fmt=".3f", center=0, cmap="RdYlGn",
                ax=ax3, linewidths=0.5, annot_kws={"size":8}, cbar_kws={"shrink":0.8})
    ax3.set_title("팩터 β 히트맵", fontsize=11, fontweight="bold")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=25, fontsize=8)
    fig.suptitle("매크로 팩터 분석  v4.3  |  6-Factor  |  자산군별 공분산 블렌딩",
                 fontsize=14, fontweight="bold")
    _savefig(fig, "fig1_macro_analysis.png")

def plot_cov_blend_comparison(S_ewma, S_factor, S_blend, assets,
                              regression_results):
    """Figure 1b: 공분산 블렌딩 전후 변동성 비교 + R² 가중치 (v4.4)"""
    print("\n[Fig 1b] 공분산 블렌딩 비교 (R² 자동 가중)...")
    vol_ewma   = np.sqrt(np.diag(S_ewma.values))   * 100
    vol_factor = np.sqrt(np.diag(S_factor.values)) * 100
    vol_blend  = np.sqrt(np.diag(S_blend.values))  * 100

    r2_arr = np.array([
        float(np.clip(regression_results.get(a, {}).get("r2", 0), 0.0, 1.0))
        for a in assets
    ])

    x = np.arange(len(assets)); w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax = axes[0]
    ax.bar(x-w, vol_ewma,   w, label="EWMA (역사적)",   color="#3498db", alpha=0.85, edgecolor="white")
    ax.bar(x,   vol_factor, w, label="Factor (β 기반)", color="#e67e22", alpha=0.85, edgecolor="white")
    ax.bar(x+w, vol_blend,  w, label="Blend (최종)",    color="#9b59b6", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(assets, fontsize=9)
    ax.set_ylabel("연간 변동성 (%)"); ax.set_title("자산별 변동성 비교", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    ax2 = axes[1]
    colors2 = ["#2ecc71" if r2 > 0.2 else "#e74c3c" for r2 in r2_arr]
    bars2 = ax2.bar(assets, r2_arr * 100, color=colors2, alpha=0.85, edgecolor="white")
    ax2.axhline(20, color="red",   lw=1.2, ls="--", alpha=0.7, label="R²=0.20 기준선")
    ax2.axhline(50, color="black", lw=0.8, ls="--", alpha=0.4, label="R²=0.50")
    ax2.set_ylabel("R² (%) — 공분산 블렌딩 가중치")
    ax2.set_title("자산별 R² (팩터 공분산 가중치 자동 결정)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25, axis="y")
    for bar, r2 in zip(bars2, r2_arr):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{r2*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("공분산 블렌딩 v4.4  |  R² 자동 가중  (역사적 EWMA + 팩터 β 기반)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig1b_cov_blend.png")

def plot_scenario_returns_by_horizon(all_horizon_returns, scenarios):
    assets = PORTFOLIO_ASSETS
    for fi, H in enumerate(SCENARIO_HORIZONS):
        print(f"\n[Fig {fi+2}] H={H}M 시나리오 수익률...")
        fig, axes = plt.subplots(1, len(scenarios), figsize=(16,6))
        if len(scenarios)==1: axes = [axes]
        for ax, sc_name in zip(axes, scenarios.keys()):
            sc   = scenarios[sc_name]
            rets = all_horizon_returns[H][sc_name]
            vals = [rets.get(a,0)*100 for a in assets]
            bcolors = [SC_COLORS[sc_name] if v>=0 else "#7f8c8d" for v in vals]
            bars = ax.bar(assets, vals, color=bcolors, alpha=0.85, edgecolor="white")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{sc_name}  (p={sc['prob']*100:.0f}%)\n{sc['description'][:30]}",
                         fontsize=10, fontweight="bold", color=SC_COLORS[sc_name])
            ax.set_ylabel("기대수익률 (%)" if ax==axes[0] else "")
            ax.tick_params(axis="x", rotation=35); ax.grid(True, alpha=0.25, axis="y")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+(0.3 if val>=0 else -0.8),
                        f"{val:+.1f}%", ha="center", fontsize=7.5)
        fig.suptitle(f"시나리오별 기대수익률  H={H}M", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _savefig(fig, f"fig{fi+2}_scenario_returns_H{H}M.png")

def plot_three_portfolios(sc_results, blended_result, scenarios, Pi):
    print("\n[Fig 4] 3개 포트폴리오 비교...")
    blended_w, blended_perf, mu_blend, _ = blended_result
    p_on  = SCENARIO_PROBS["Risk-On"]; p_off = SCENARIO_PROBS["Risk-Off"]
    all_portfolios = {
        "Risk-On":  (sc_results["Risk-On"]["weights"],  sc_results["Risk-On"]["perf"],  "#2ecc71"),
        "Risk-Off": (sc_results["Risk-Off"]["weights"], sc_results["Risk-Off"]["perf"], "#e74c3c"),
        f"블렌딩\n(On={p_on*100:.0f}%·Off={p_off*100:.0f}%)":
                    (blended_w, blended_perf, "#9b59b6"),
    }
    port_names = list(all_portfolios.keys()); n_port = len(port_names)
    assets = PORTFOLIO_ASSETS; colors_list = ["#2ecc71","#e74c3c","#9b59b6"]
    fig = plt.figure(figsize=(24,18))
    gs  = gridspec.GridSpec(3, n_port, figure=fig, hspace=0.55, wspace=0.35)
    for col, (pname, (w, perf, color)) in enumerate(all_portfolios.items()):
        ax = fig.add_subplot(gs[0, col])
        nz = {k: v for k,v in w.items() if v>0.005}
        bond_w = sum(w.get(a,0) for a in assets if ASSETS[a]["type"].startswith("bond"))
        ax.pie(nz.values(), labels=nz.keys(), autopct="%1.1f%%",
               colors=sns.color_palette("husl", len(nz)), startangle=90,
               explode=[0.04]*len(nz), textprops={"fontsize":8},
               wedgeprops={"linewidth":0.8, "edgecolor":"white"})
        ax.set_title(f"{pname}\nSharpe={perf[2]:.3f}  μ={perf[0]*100:.2f}%  σ={perf[1]*100:.2f}%\n채권={bond_w*100:.1f}%",
                     fontsize=9, fontweight="bold", color=color)
    ax_bar = fig.add_subplot(gs[1,:])
    x = np.arange(len(assets)); w_grp = 0.6/n_port
    for i, (pname, (w, perf, color)) in enumerate(all_portfolios.items()):
        vals = [w.get(a,0)*100 for a in assets]
        off  = (i-n_port/2+0.5)*w_grp
        ax_bar.bar(x+off, vals, w_grp*0.9, label=pname.replace("\n"," "),
                   color=color, alpha=0.85, edgecolor="white")
    for a in assets:
        if ASSETS[a]["type"].startswith("bond"):
            bi = assets.index(a)
            ax_bar.axvspan(bi-0.5, bi+0.5, alpha=0.06, color="#3498db")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(assets, fontsize=10)
    ax_bar.set_ylabel("비중 (%)"); ax_bar.set_title("3개 포트폴리오 비중 비교  (파란 배경=채권)", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=10); ax_bar.grid(True, alpha=0.25, axis="y")
    ax_sr = fig.add_subplot(gs[2,:2])
    sr_vals = [all_portfolios[pn][1][2] for pn in port_names]
    x3 = np.arange(n_port)
    b1 = ax_sr.bar(x3, sr_vals, 0.5, color=colors_list, alpha=0.85, edgecolor="white")
    for bar, val in zip(b1, sr_vals):
        ax_sr.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                   ha="center", fontsize=10, fontweight="bold")
    ax_sr.set_xticks(x3); ax_sr.set_xticklabels([pn.replace("\n"," ") for pn in port_names], fontsize=9)
    ax_sr.set_title("Sharpe 비율 비교", fontsize=12, fontweight="bold")
    ax_sr.set_ylabel("Sharpe"); ax_sr.grid(True, alpha=0.25, axis="y")
    ax_tbl = fig.add_subplot(gs[2,2]); ax_tbl.axis("off")
    rows = [["포트폴리오","μ","σ","Sharpe","채권"]]
    for pname, (w, perf, color) in all_portfolios.items():
        bw = sum(w.get(a,0) for a in assets if ASSETS[a]["type"].startswith("bond"))
        rows.append([pname.replace("\n"," "), f"{perf[0]*100:.2f}%",
                     f"{perf[1]*100:.2f}%", f"{perf[2]:.3f}", f"{bw*100:.1f}%"])
    tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center", colWidths=[0.38,0.15,0.15,0.15,0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1,1.9)
    for j in range(5): tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i, color in enumerate(colors_list):
        for j in range(5): tbl[i+1,j].set_facecolor(color+"28")
    fig.suptitle(
        f"3개 포트폴리오 비교  v4.3\n"
        f"Risk-On p={p_on*100:.0f}%  Risk-Off p={p_off*100:.0f}%  (직접 입력)\n"
        f"공분산: EWMA+팩터 자산군별 블렌딩 | MDD_wmax | MIN_ALL={MIN_WEIGHT_ALL*100:.0f}%",
        fontsize=12, fontweight="bold")
    _savefig(fig, "fig4_three_portfolios.png")

def plot_efficient_frontiers(sc_results, blended_result, scenarios, prices):
    print("\n[Fig 5] 효율적 프론티어...")
    assets = PORTFOLIO_ASSETS; rng_s = np.random.default_rng(42)
    blended_w, blended_perf, mu_blend, S_blend = blended_result
    p_on = SCENARIO_PROBS["Risk-On"]; p_off = SCENARIO_PROBS["Risk-Off"]
    fig, axes = plt.subplots(1,3, figsize=(22,7))
    all_items = list(sc_results.items()) + [
        (f"블렌딩(On={p_on*100:.0f}%·Off={p_off*100:.0f}%)",
         {"mu_bl": mu_blend, "S_bl": S_blend, "weights": blended_w, "perf": blended_perf})
    ]
    plot_colors = ["#2ecc71","#e74c3c","#9b59b6"]
    for ax, (sc_name, res), color in zip(axes, all_items, plot_colors):
        mu_arr = np.array([res["mu_bl"].get(a,0) if isinstance(res["mu_bl"], pd.Series)
                           else res["mu_bl"][a] for a in assets])
        S_arr = res["S_bl"].values
        rs, vs, srs = [], [], []
        for _ in range(4000):
            ww = rng_s.dirichlet(np.ones(len(assets)))
            rr = ww@mu_arr; vv = np.sqrt(ww@S_arr@ww)
            rs.append(rr); vs.append(vv)
            srs.append((rr-RISK_FREE_RATE_6M)/vv if vv>0 else 0)
        sc7 = ax.scatter(vs, rs, c=srs, cmap="viridis", s=6, alpha=0.45, vmin=-0.5, vmax=0.8)
        plt.colorbar(sc7, ax=ax, label="Sharpe", shrink=0.8)
        w_opt = np.array([res["weights"].get(a,0) for a in assets]); perf = res["perf"]
        ax.scatter(np.sqrt(w_opt@S_arr@w_opt), w_opt@mu_arr, color=color, s=350,
                   zorder=6, marker="*", edgecolors="black", linewidths=0.8,
                   label=f"최적 SR={perf[2]:.3f}")
        ax.set_xlabel("변동성 (σ)"); ax.set_ylabel("기대수익률 (μ)" if ax==axes[0] else "")
        ax.set_title(f"{sc_name}\nSR={perf[2]:.3f}  μ={perf[0]*100:.2f}%  σ={perf[1]*100:.2f}%",
                     fontsize=9, fontweight="bold", color=color)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.suptitle("효율적 프론티어  (Monte Carlo N=4,000)  |  3개 포트폴리오", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig5_efficient_frontiers.png")

def plot_dns(dns_fc_df, dns_fc_yields, dns_filt_factors, yields_df):
    if dns_fc_df is None or dns_fc_yields is None: return
    print("\n[Fig 6] DNS 금리커브...")
    mats   = sorted(dns_fc_yields)
    fc_col = ["#3498db","#e67e22","#2ecc71","#9b59b6","#e74c3c","#1abc9c","#f39c12"]
    fig, axes = plt.subplots(2,2, figsize=(16,10)); axes = axes.flatten()
    ax_a = axes[0]
    if dns_filt_factors is not None:
        for col,color,lbl in zip(["L","S","C"],["#3498db","#e74c3c","#2ecc71"],["Level","Slope","Curvature"]):
            if col in dns_filt_factors.columns:
                ax_a.plot(dns_filt_factors.index, dns_filt_factors[col], label=lbl, color=color, lw=1.5)
        ax_a.axhline(0,color="k",lw=0.7,ls="--",alpha=0.4)
        ax_a.set_title("Kalman DNS 인자", fontsize=11, fontweight="bold")
        ax_a.legend(fontsize=9); ax_a.grid(True, alpha=0.25)
    ax_b = axes[1]
    if yields_df is not None and not yields_df.empty:
        curr = yields_df.iloc[-1]
        ax_b.plot(curr.index.astype(float), curr.values, "o-", color="#2c3e50", lw=2, markersize=7,
                  label=f"현재 ({yields_df.index[-1].strftime('%Y-%m')})")
    fc_vals = [dns_fc_yields[m] for m in mats]
    ax_b.plot(mats, fc_vals, "s--", color="#e74c3c", lw=2, markersize=7, label="12M 예측")
    ax_b.fill_between(mats,[v-0.3 for v in fc_vals],[v+0.3 for v in fc_vals],color="#e74c3c",alpha=0.08)
    ax_b.set_title("현재 vs 12M 예측 커브", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("만기 (년)"); ax_b.set_ylabel("금리 (%)"); ax_b.legend(fontsize=9); ax_b.grid(True,alpha=0.25)
    ax_b.set_xticks(mats); ax_b.set_xticklabels([f"{m}Y" for m in mats], fontsize=8)
    try:
        fc_idx = pd.date_range(pd.Timestamp.today()+pd.offsets.MonthEnd(1), periods=len(dns_fc_df), freq="ME")
    except Exception:
        fc_idx = pd.date_range(pd.Timestamp.today()+pd.offsets.MonthEnd(1), periods=len(dns_fc_df), freq="M")
    fc_paths = {}
    for i, row in dns_fc_df.iterrows():
        for m in mats:
            fc_paths.setdefault(m,[]).append(float(_ns_yield_from_factors(m,row["L"],row["S"],row["C"])))
    for j, m in enumerate(mats):
        axes[2].plot(fc_idx, fc_paths[m], label=f"{m}Y", color=fc_col[j%len(fc_col)], lw=1.6)
    axes[2].set_title("만기별 금리 예측 경로 (12M)", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=8, ncol=2); axes[2].grid(True, alpha=0.25)
    axes[2].tick_params(axis="x", rotation=25)
    for col,color,lbl in zip(["L","S","C"],["#3498db","#e74c3c","#2ecc71"],["Level","Slope","Curvature"]):
        if col in dns_fc_df.columns:
            axes[3].plot(fc_idx, dns_fc_df[col].values, label=lbl, color=color, lw=2, marker="o", markersize=4)
    axes[3].axhline(0,color="k",lw=0.7,ls="--",alpha=0.4)
    axes[3].set_title("DNS 인자 예측 경로", fontsize=11, fontweight="bold")
    axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.25)
    axes[3].tick_params(axis="x", rotation=25)
    fig.suptitle("Dynamic Nelson-Siegel + Kalman Filter  —  금리커브 예측  (v4.2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig6_dns_forecast.png")


# ============================================================
# ■ STEP 11: 위험지표 계산 — MDD / VaR / ES
# ============================================================
def compute_risk_metrics(prices, assets, lookback=None,
                         var_levels=(0.95, 0.99)):
    """
    자산별 역사적 위험지표 산출.

    MDD   : 최대낙폭 (전체 기간 또는 lookback)
    VaR   : 역사적 분위수법 (신뢰수준 95%, 99%)
            VaR_α = -percentile(r, 1-α) × √12  (연환산, 양수)
    ES    : 조건부 기대손실 (Expected Shortfall / CVaR)
            ES_α  = -mean(r | r ≤ -VaR_α_월)   (연환산, 양수)

    반환: DataFrame (index=asset, columns=MDD/VaR95/VaR99/ES95/ES99)
    """
    try:   mo = prices.resample("ME").last()
    except: mo = prices.resample("M").last()

    rets_mo = mo.pct_change().dropna()
    if lookback:
        rets_mo = rets_mo.iloc[-lookback:]
        mo      = mo.iloc[-lookback:]

    records = {}
    print(f"\n[위험지표] {rets_mo.index[0].date()}~{rets_mo.index[-1].date()}")
    print(f"  {'자산':>5s}  {'MDD':>7s}  "
          f"{'VaR95':>7s}  {'VaR99':>7s}  "
          f"{'ES95':>6s}  {'ES99':>6s}")

    for a in assets:
        if a not in mo.columns:
            records[a] = {"MDD": np.nan,"VaR95": np.nan,"VaR99": np.nan,
                          "ES95": np.nan,"ES99": np.nan}
            continue
        px  = mo[a].dropna()
        r   = rets_mo[a].dropna().values  # 월간 수익률 (소수점)

        # MDD
        cum_max = np.maximum.accumulate(px.values)
        dd      = (px.values - cum_max) / cum_max
        mdd     = float(dd.min())

        # VaR & ES (연환산)
        row = {"MDD": mdd}
        for lvl in var_levels:
            key = int(lvl * 100)
            cutoff = np.percentile(r, (1 - lvl) * 100)   # 음수
            var_ann = -cutoff * np.sqrt(12)               # 연환산, 양수
            tail    = r[r <= cutoff]
            es_ann  = (-tail.mean() * np.sqrt(12)) if len(tail) > 0 else var_ann
            row[f"VaR{key}"] = float(var_ann)
            row[f"ES{key}"]  = float(es_ann)

        records[a] = row
        print(f"  {a:>5s}  {mdd*100:>6.1f}%  "
              f"{row['VaR95']*100:>6.1f}%  {row['VaR99']*100:>6.1f}%  "
              f"{row['ES95']*100:>5.1f}%  {row['ES99']*100:>5.1f}%")

    return pd.DataFrame(records).T


def plot_risk_constraints(prices, sc_results, blended_result,
                          regression_results):
    """
    [Fig 7] 비중 제약 및 위험지표 통합 시각화.

    행 1-L: 자산별 MDD / VaR95 / ES95 비교 막대
    행 1-R: MDD → w_max 역산 결과 + 실제 적용 비중 상한
    행 2:   시나리오별 + 블렌딩 포트폴리오 비중 vs 비중 상한 비교
    행 3:   Risk-On / Risk-Off / 블렌딩 포트폴리오 위험지표 비교
    """
    print("\n[Fig 7] 비중 제약 및 위험지표 시각화...")
    assets = PORTFOLIO_ASSETS
    n      = len(assets)
    x      = np.arange(n)

    # ── 위험지표 산출 ──────────────────────────────────────
    risk_df = compute_risk_metrics(prices, assets, MDD_LOOKBACK)

    # ── MDD→w_max 재계산 (시각화용) ───────────────────────
    mdd_abs    = risk_df["MDD"].abs()
    median_mdd = float(mdd_abs.median())
    wmax_list  = []
    for a in assets:
        raw  = MDD_BASE_MAX * (median_mdd / max(float(mdd_abs[a]), 1e-6))
        wmax = float(np.clip(raw, MDD_WMAX_FLOOR, MDD_WMAX_CAP))
        wmax = max(wmax, MIN_WEIGHT_ALL + 0.001)
        wmax_list.append(wmax)

    # ── 포트폴리오 실제 비중 ───────────────────────────────
    blended_w = blended_result[0]
    port_weights = {
        "Risk-On":  sc_results["Risk-On"]["weights"],
        "Risk-Off": sc_results["Risk-Off"]["weights"],
        "블렌딩":   blended_w,
    }
    port_colors = {"Risk-On": "#2ecc71", "Risk-Off": "#e74c3c", "블렌딩": "#9b59b6"}

    fig = plt.figure(figsize=(24, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── 행1-L: 자산별 MDD/VaR95/ES95 ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bw  = 0.22
    mdd_v  = risk_df["MDD"].reindex(assets).abs().values * 100
    var_v  = risk_df["VaR95"].reindex(assets).values * 100
    es_v   = risk_df["ES95"].reindex(assets).values * 100
    ax1.bar(x - bw, mdd_v, bw, label="MDD (%)",   color="#e74c3c", alpha=0.80, edgecolor="white")
    ax1.bar(x,      var_v, bw, label="VaR 95% (연환산)", color="#f39c12", alpha=0.80, edgecolor="white")
    ax1.bar(x + bw, es_v,  bw, label="ES 95% (연환산)",  color="#9b59b6", alpha=0.80, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(assets, fontsize=9)
    ax1.set_ylabel("위험지표 (%)"); ax1.set_title("자산별 위험지표 (MDD / VaR95 / ES95)",
                                                 fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25, axis="y")
    for i, (m, v, e) in enumerate(zip(mdd_v, var_v, es_v)):
        ax1.text(i-bw, m+0.3, f"{m:.0f}", ha="center", fontsize=6.5, color="#e74c3c")
        ax1.text(i,    v+0.3, f"{v:.0f}", ha="center", fontsize=6.5, color="#f39c12")
        ax1.text(i+bw, e+0.3, f"{e:.0f}", ha="center", fontsize=6.5, color="#9b59b6")

    # ── 행1-R: MDD→w_max 역산 + 최소·최대 비중 라인 ───────
    ax2 = fig.add_subplot(gs[0, 1])
    bar_c = ["#3498db" if ASSETS[a]["type"].startswith("bond") else "#2ecc71"
             for a in assets]
    bars2 = ax2.bar(x, [w*100 for w in wmax_list], color=bar_c, alpha=0.75, edgecolor="white",
                    label="w_max (MDD 역산)")
    ax2.axhline(MIN_WEIGHT_ALL*100, color="orange", lw=1.5, ls="--", label=f"최소비중 {MIN_WEIGHT_ALL*100:.0f}%")
    ax2.axhline(BOND_TOTAL_MIN*100 / 3, color="#3498db", lw=1.2, ls=":",
                label=f"채권 개별 최소 ({BOND_TOTAL_MIN*100:.0f}%÷3={BOND_TOTAL_MIN/3*100:.1f}%)")
    ax2.axhline(MDD_WMAX_CAP*100, color="red", lw=1, ls="--", alpha=0.5,
                label=f"w_max 상한 {MDD_WMAX_CAP*100:.0f}%")
    ax2.set_xticks(x); ax2.set_xticklabels(assets, fontsize=9)
    ax2.set_ylabel("비중 상한 (%)"); ax2.set_title("MDD 기반 동적 비중 상한 (w_max)",
                                                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25, axis="y")
    for bar, wm in zip(bars2, wmax_list):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{wm*100:.0f}%", ha="center", fontsize=8, fontweight="bold")

    # ── 행2: 포트폴리오 실제 비중 vs w_max ────────────────
    ax3 = fig.add_subplot(gs[1, :])
    grp_w = 0.18; positions = {pn: x + (i-1)*grp_w
                                for i, pn in enumerate(port_weights.keys())}
    for pn, pw in port_weights.items():
        vals = [pw.get(a, 0)*100 for a in assets]
        ax3.bar(positions[pn], vals, grp_w*0.9,
                label=pn, color=port_colors[pn], alpha=0.80, edgecolor="white")

    # w_max 라인
    ax3.step(np.arange(-0.5, n+0.5), [wmax_list[0]]*1 + [w*100 for w in wmax_list] + [wmax_list[-1]]*0,
             color="black", lw=1.5, ls="--", where="mid", label="w_max 상한")
    ax3.axhline(MIN_WEIGHT_ALL*100, color="orange", lw=1.2, ls=":", label=f"최소비중 {MIN_WEIGHT_ALL*100:.0f}%")

    for a in assets:
        if ASSETS[a]["type"].startswith("bond"):
            bi = assets.index(a)
            ax3.axvspan(bi-0.45, bi+0.45, alpha=0.05, color="#3498db")

    ax3.set_xticks(x); ax3.set_xticklabels(assets, fontsize=10)
    ax3.set_ylabel("포트폴리오 비중 (%)")
    ax3.set_title("시나리오별 실제 비중 vs 비중 상한 (파란 배경=채권)",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, ncol=4); ax3.grid(True, alpha=0.25, axis="y")

    # ── 행3: 포트폴리오 위험지표 비교 ─────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    # 시뮬레이션 기반 포트폴리오 MDD / VaR95 / ES95
    try:   mo_p = prices.resample("ME").last()
    except: mo_p = prices.resample("M").last()
    mo_p = mo_p[assets].dropna()
    r_mo = mo_p.pct_change().dropna()

    port_risk_rows = []
    for pn, pw in port_weights.items():
        w_arr = np.array([pw.get(a, 0) for a in assets])
        w_arr = w_arr / w_arr.sum()
        port_r = (r_mo * w_arr).sum(axis=1).values

        # MDD
        cum = np.cumprod(1 + port_r)
        peak = np.maximum.accumulate(cum)
        mdd_p = float(((cum - peak) / peak).min()) * 100

        # VaR/ES 95
        cut = np.percentile(port_r, 5)
        var_p  = -cut * np.sqrt(12) * 100
        tail_p = port_r[port_r <= cut]
        es_p   = (-tail_p.mean() * np.sqrt(12) * 100) if len(tail_p) > 0 else var_p

        port_risk_rows.append({"포트폴리오": pn,
                                "MDD":   abs(mdd_p),
                                "VaR95": var_p,
                                "ES95":  es_p})

    port_risk_df = pd.DataFrame(port_risk_rows).set_index("포트폴리오")
    px4 = np.arange(len(port_risk_df))
    bw4 = 0.22
    ax4.bar(px4 - bw4, port_risk_df["MDD"].values,   bw4,
            color="#e74c3c", alpha=0.80, edgecolor="white", label="포트폴리오 MDD (%)")
    ax4.bar(px4,        port_risk_df["VaR95"].values, bw4,
            color="#f39c12", alpha=0.80, edgecolor="white", label="포트폴리오 VaR95 (연환산)")
    ax4.bar(px4 + bw4, port_risk_df["ES95"].values,  bw4,
            color="#9b59b6", alpha=0.80, edgecolor="white", label="포트폴리오 ES95 (연환산)")

    for i, row in enumerate(port_risk_df.itertuples()):
        for offset, val, col in [(-bw4, row.MDD, "#e74c3c"),
                                  (0,   row.VaR95, "#f39c12"),
                                  (bw4, row.ES95,  "#9b59b6")]:
            ax4.text(i + offset, val + 0.2, f"{val:.1f}%",
                     ha="center", fontsize=9, fontweight="bold", color=col)

    ax4.set_xticks(px4)
    ax4.set_xticklabels([pn + f"\n({list(port_colors.values())[i]})"
                         for i, pn in enumerate(port_risk_df.index)], fontsize=10)
    ax4.set_ylabel("위험지표 (%)")
    ax4.set_title("포트폴리오별 위험지표 비교 (MDD / VaR95 / ES95, 역사적 월간 수익률 기반)",
                  fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "포트폴리오 비중 제약 및 위험지표  v4.3\n"
        f"비중 제약: MIN={MIN_WEIGHT_ALL*100:.0f}%  채권합계≥{BOND_TOTAL_MIN*100:.0f}%  "
        f"MDD기반 동적w_max[{MDD_WMAX_FLOOR*100:.0f}%~{MDD_WMAX_CAP*100:.0f}%]",
        fontsize=12, fontweight="bold")
    _savefig(fig, "fig7_risk_constraints.png")


# ============================================================
# ■ MAIN
# ============================================================
def main():
    p_on  = SCENARIO_PROBS["Risk-On"]
    p_off = SCENARIO_PROBS["Risk-Off"]
    print("="*70)
    print("  Macro-Driven BL Pipeline  v4.3")
    print("  시나리오: Risk-On / Risk-Off")
    print(f"  입력 확률: Risk-On={p_on*100:.0f}%  Risk-Off={p_off*100:.0f}%")
    print(f"  수익률 블렌딩: {'R² 자동 가중 (Prior=mu_hist+R²×alpha)' if USE_FACTOR_RETURN else '순수 mu_hist'}")
    print(f"  공분산 블렌딩: {'R² 자동 가중 (W_ij=(R²_i+R²_j)/2)' if USE_FACTOR_COV else '순수 EWMA'}")
    print("="*70)

    assert abs(sum(SCENARIO_PROBS.values())-1.0) < 1e-6, \
        f"SCENARIO_PROBS 합계 != 1: {sum(SCENARIO_PROBS.values()):.4f}"

    SCENARIOS["Risk-On"]["prob"]  = p_on
    SCENARIOS["Risk-Off"]["prob"] = p_off

    prices   = fetch_asset_data()
    macro_df = fetch_macro_data()

    reg_returns, means, port_rets_1m = _prepare_returns(prices, macro_df)
    port_rets_asset = reg_returns[PORTFOLIO_ASSETS].dropna(how="all")
    reg_results          = run_regressions(port_rets_asset, macro_df)
    diag_summary, vif_df = run_regression_diagnostics(reg_results, macro_df)

    dns_fc_df = dns_fc_yields = yields_df = dns_filt_factors = None
    try:
        yields_df, mats = fetch_treasury_yields()
        if yields_df is not None and len(yields_df) > 24:
            dns_fc_df, dns_fc_yields, dns_filt_factors, _, _ = \
                dns_kalman_forecast(yields_df, mats)
    except Exception as e:
        print(f"  ⚠ DNS 실패: {e}")

    scenario_alpha_annual = compute_scenario_alpha_annual(reg_results, SCENARIOS)
    print("\n[Step 6b] 시각화용 수익률...")
    all_horizon_returns = compute_scenario_returns_for_viz(reg_results, SCENARIOS)

    try:   mo_1m = prices.resample("ME").last()
    except: mo_1m = prices.resample("M").last()
    monthly_1m = mo_1m.pct_change().dropna() * 100

    sc_results, Pi, S = run_all_scenario_optimizations(
        prices, reg_results, scenario_alpha_annual, SCENARIOS,
        port_rets_1m=monthly_1m[PORTFOLIO_ASSETS].dropna(how="all"),
        dns_fc_df=dns_fc_df, dns_filt_factors=dns_filt_factors,
        macro_df=macro_df,
    )

    blended_result = compute_blended_portfolio(sc_results, SCENARIO_PROBS, prices)
    blended_w, blended_perf, mu_blend, S_blend = blended_result

    latest = get_latest_prices(prices[PORTFOLIO_ASSETS])
    da = DiscreteAllocation(blended_w, latest, total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
    blended_alloc, blended_leftover = da.greedy_portfolio()

    print("\n"+"="*70)
    print("  최종 포트폴리오 요약  (3개)")
    print("="*70)
    portfolios = [
        ("Risk-On",  sc_results["Risk-On"]["weights"],  sc_results["Risk-On"]["perf"],
         sc_results["Risk-On"]["alloc"],  sc_results["Risk-On"]["leftover"]),
        ("Risk-Off", sc_results["Risk-Off"]["weights"], sc_results["Risk-Off"]["perf"],
         sc_results["Risk-Off"]["alloc"], sc_results["Risk-Off"]["leftover"]),
        (f"블렌딩(On={p_on*100:.0f}%·Off={p_off*100:.0f}%)",
         blended_w, blended_perf, blended_alloc, blended_leftover),
    ]
    for pname, w, perf, alloc, leftover in portfolios:
        bw = sum(w.get(a,0) for a in PORTFOLIO_ASSETS if ASSETS[a]["type"].startswith("bond"))
        print(f"\n  [{pname}]  μ={perf[0]*100:.2f}%  σ={perf[1]*100:.2f}%  Sharpe={perf[2]:.3f}  채권={bw*100:.1f}%")
        for t in sorted(alloc, key=alloc.get, reverse=True):
            print(f"    {t:>5s}: {alloc[t]:>4d}주")
        print(f"    잔여 현금: ${leftover:,.2f}")

    # ── 시각화 ──────────────────────────────────────────────
    # Fig 0: 기대수익률 구성 분해 (역사적 / alpha 기여 / Prior / 직접입력)
    plot_return_decomposition(
        monthly_1m[PORTFOLIO_ASSETS].dropna(how="all"),
        scenario_alpha_annual,   # DNS 보정 전 원본 (시각화용)
        sc_results,
        reg_results,
    )
    plot_macro_analysis(macro_df, reg_results)

    # 공분산 블렌딩 비교 차트 (v4.3 신규, USE_FACTOR_COV=True일 때만)
    if USE_FACTOR_COV:
        try:
            assets = PORTFOLIO_ASSETS
            S_ewma_viz   = build_ewma_covariance(prices, assets)
            S_factor_viz = build_factor_covariance(reg_results, macro_df, assets)
            S_blend_viz  = build_blended_covariance(
                S_ewma_viz, S_factor_viz, assets, reg_results)
            plot_cov_blend_comparison(
                S_ewma_viz, S_factor_viz, S_blend_viz, assets, reg_results)
        except Exception as e:
            print(f"  ⚠ 공분산 비교 차트 실패({e})")

    plot_scenario_returns_by_horizon(all_horizon_returns, SCENARIOS)
    plot_three_portfolios(sc_results, blended_result, SCENARIOS, Pi)
    plot_efficient_frontiers(sc_results, blended_result, SCENARIOS, prices)
    plot_dns(dns_fc_df, dns_fc_yields, dns_filt_factors, yields_df)

    # Fig 7: 비중 제약 및 위험지표
    plot_risk_constraints(prices, sc_results, blended_result, reg_results)

    print("\n"+"="*70)
    print("  Pipeline Complete ✓  (v4.3)")
    print(f"  출력: fig0~7 PNG → {OUTPUT_DIR}")
    print("="*70)

    return {
        "prices":     prices,
        "macro":      macro_df,
        "regression": reg_results,
        "sc_results": sc_results,
        "blended":    {"weights": blended_w, "perf": blended_perf,
                       "alloc": blended_alloc, "leftover": blended_leftover},
        "Pi":         Pi,
    }


if __name__ == "__main__":
    results = main()