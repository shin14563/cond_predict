# -*- coding: utf-8 -*-
"""
pref2cond_predict.py  (出力フォーマット刷新版)
- 入力: Pref2Cond.csv（縦持ち: 1列目=パラメータ名, 2列目以降=case列）
  期待する行名:
    CaseName
    EnergyCategory_-
    lower_BPI_- , upper_BPI_-
    lower_Insulation_mm , upper_Insulation_mm
    lower_WWR_- , upper_WWR_-            # ★ 0〜1（無次元）
    lower_VerticalLength_mm , upper_VerticalLength_mm
    lower_HorizontalLength_mm, upper_HorizontalLength_mm
    lower_CoverageRatio_- , upper_CoverageRatio_-  # 0〜1（無次元）

- 出力: 各caseごとに 1ファイル（CSV）
    [ブロック1] 入力レンジのまとめ 1行
    空行
    [ブロック2] 代表8行の表
       代表が1件も見つからなければ 1行だけ: 「- No results found. -」

  代表8行の列:
    CaseName, EnergyCategory_-, InternalHeat_-,
    VerticalLength_mm, HorizontalLength_mm, CoverageRatio_-,
    Insulation_mm, WWR_-, U-value_-, SHGC_-,
    predictBEI_-, predictBPI_-, predictCooling_MJm2, predictHeating_MJm2

- モデル: app_resources/model.cbm（CatBoost MultiRMSE: ["Cooling","Heating","BEI","BPI"]）
- 壁面パラメータのサンプリング:
    ① 指定レンジに該当する PATTERNS の組合せがあれば、その出現比で配分
    ② 指定が一切なければ PATTERNS 全体から比率配分
    ③ 指定レンジに該当が0の場合、指定レンジを優先（一様サンプリング）しつつ、
       件数配分は PATTERNS 比率
"""

from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

# ---- CatBoost（推論のみ使用） ----
try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise RuntimeError(
        "CatBoost is not available. Install with: pip install catboost") from e


# ========== 実行/同梱リソースユーティリティ ==========
def executable_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", executable_dir())
    return os.path.join(base, rel)


# ========== 既定設定 ==========
DEFAULT_CONFIG = {
    "model_output_order": ["Cooling", "Heating", "BEI", "BPI"],
    "n_random": 20000,
    "bpi_tolerance": 0.02,
    "grid_step": 0.01,
    "feature_names": {
        "syoene_onehot": [f"syoene_{i}" for i in range(1, 9)],  # 1..8
        "setsubi": ["setsubi_0", "setsubi_1"],
        "cont": [
            "NS_Wall", "EW_Wall", "AreaRatio", "dannetu", "kaikouritu",
            "U-chi", "nissyasyutoku"
        ],
    },
}

# === 壁面パターン（学習データの出現カウント）===
PATTERNS = pd.DataFrame({
    "NS_Wall": [136.4616, 180.7740, 197.6436, 204.6924, 271.1808],
    "EW_Wall": [204.6924, 271.1808, 131.7492, 136.4616, 180.7740],
    "AreaRatio": [0.932252, 0.531220, 1.000000, 0.932252, 0.531220],
    "counts": [9599, 9600, 4792, 9593, 9593],
})


# ========== 小道具 ==========
def _alloc_counts_to_total(df_counts: pd.DataFrame,
                           total: int,
                           seed: int = 42) -> np.ndarray:
    """counts 比率を保ったまま total 件に丸める（端数はランダム配分）"""
    rng = np.random.default_rng(seed)
    w = df_counts["counts"].to_numpy(dtype=float)
    p = w / w.sum()
    raw = p * total
    base = np.floor(raw).astype(int)
    short = total - base.sum()
    frac = raw - base
    if short > 0:
        add_idx = rng.choice(len(base),
                             size=short,
                             replace=False,
                             p=(frac / frac.sum()) if frac.sum() > 0 else None)
        base[add_idx] += 1
    return base


def _mask_patterns_by_range(df: pd.DataFrame, ns_min, ns_max, ew_min, ew_max,
                            ar_min, ar_max) -> pd.Series:
    """範囲で PATTERNS をフィルタ"""
    m = pd.Series(True, index=df.index)
    if ns_min is not None: m &= df["NS_Wall"] >= ns_min
    if ns_max is not None: m &= df["NS_Wall"] <= ns_max
    if ew_min is not None: m &= df["EW_Wall"] >= ew_min
    if ew_max is not None: m &= df["EW_Wall"] <= ew_max
    if ar_min is not None: m &= df["AreaRatio"] >= ar_min
    if ar_max is not None: m &= df["AreaRatio"] <= ar_max
    return m


def _sample_walls_case_aware(
        n: int,
        rng: np.random.Generator,
        ns_range: tuple[float | None, float | None],
        ew_range: tuple[float | None, float | None],
        ar_range: tuple[float | None, float | None],
        seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定レンジ優先で NS/EW/AreaRatio を n 件生成（PATTERNS の比率を維持）
    """
    ns_min, ns_max = ns_range
    ew_min, ew_max = ew_range
    ar_min, ar_max = ar_range

    any_constraint = any(
        v is not None
        for v in [ns_min, ns_max, ew_min, ew_max, ar_min, ar_max])
    pats = (PATTERNS[_mask_patterns_by_range(PATTERNS, ns_min, ns_max, ew_min,
                                             ew_max, ar_min, ar_max)].copy()
            if any_constraint else PATTERNS.copy())
    pats = pats.reset_index(drop=True)

    if len(pats) > 0:
        alloc = _alloc_counts_to_total(pats, total=n, seed=seed)
        NS = np.empty(n, dtype=float)
        EW = np.empty(n, dtype=float)
        AR = np.empty(n, dtype=float)
        pos = 0
        for k in range(len(pats)):
            row = pats.iloc[k]
            cnt = int(alloc[k])
            if cnt <= 0:
                continue
            NS[pos:pos + cnt] = float(row["NS_Wall"])
            EW[pos:pos + cnt] = float(row["EW_Wall"])
            AR[pos:pos + cnt] = float(row["AreaRatio"])
            pos += cnt
        idx = rng.permutation(n)
        return NS[idx], EW[idx], AR[idx]

    # ③ 指定レンジに該当0件 → 指定レンジを優先（範囲から一様）しつつ PATTERNS 比率で配分
    def _minmax(col: str, vmin, vmax):
        base_min = float(PATTERNS[col].min())
        base_max = float(PATTERNS[col].max())
        a = base_min if vmin is None else float(vmin)
        b = base_max if vmax is None else float(vmax)
        if a > b: a, b = b, a
        return a, b

    ns_lo, ns_hi = _minmax("NS_Wall", ns_min, ns_max)
    ew_lo, ew_hi = _minmax("EW_Wall", ew_min, ew_max)
    ar_lo, ar_hi = _minmax("AreaRatio", ar_min, ar_max)

    alloc_all = _alloc_counts_to_total(PATTERNS, total=n, seed=seed)
    NS = np.empty(n, dtype=float)
    EW = np.empty(n, dtype=float)
    AR = np.empty(n, dtype=float)
    pos = 0
    for k in range(len(PATTERNS)):
        cnt = int(alloc_all[k])
        if cnt <= 0:
            continue
        NS[pos:pos + cnt] = rng.uniform(ns_lo, ns_hi, size=cnt)
        EW[pos:pos + cnt] = rng.uniform(ew_lo, ew_hi, size=cnt)
        AR[pos:pos + cnt] = rng.uniform(ar_lo, ar_hi, size=cnt)
        pos += cnt

    idx = rng.permutation(n)
    return NS[idx], EW[idx], AR[idx]


# ========== YAML ローダ（任意） ==========
def load_yaml_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        warnings.warn(f"Failed to read yaml '{path}': {e}")
        return {}


# ========== モデル読み込み ==========
def load_model(model_path: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


# ========== Pref2Cond.csv の読み取り（縦持ち） ==========
def read_pref2cond_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df[first_col] = (df[first_col].astype(str).str.strip().str.replace(
        r"\s+", "", regex=True).str.replace("　", "", regex=False))
    new_cols = [df.columns[0]]
    for c in df.columns[1:]:
        nc = str(c).strip().replace("　", "").replace(" ", "")
        if nc == "" or nc.lower() == "nan":
            nc = "case"
        new_cols.append(nc)
    df.columns = new_cols
    return df


# ========== ケース辞書へ ==========
def parse_cases(df: pd.DataFrame) -> dict[str, dict]:
    """縦持ち → {case: params}"""
    first = df.columns[0]
    cases: dict[str, dict] = {}
    keys = df[first].tolist()

    for c in df.columns[1:]:
        params = {}
        for k, v in zip(keys, df[c].tolist()):
            if pd.isna(v):
                continue
            # 期待キーはそのまま保存（CaseName, EnergyCategory_-, lower_*, upper_*）
            try:
                params[k] = float(v) if k != "CaseName" else str(v).strip()
            except Exception:
                params[k] = str(v).strip()
        cases[c] = params
    return cases


# ========== サンプリングユーティリティ ==========
def _grid_choice(low: float, high: float, step: float, size: int,
                 rng: np.random.Generator) -> np.ndarray:
    low, high = float(low), float(high)
    if high < low:
        low, high = high, low
    grid = np.arange(low, high + 1e-12, step)
    if len(grid) == 0:
        grid = np.array([low])
    return rng.choice(grid, size=size, replace=True)


def _int_choice(low: int, high: int, size: int,
                rng: np.random.Generator) -> np.ndarray:
    if high < low:
        low, high = high, low
    return rng.integers(low, high + 1, size=size, endpoint=True)


# ========== 候補生成 ==========
def make_candidates(case_params: dict, cfg: dict, feature_order: list[str],
                    n_random: int, rng: np.random.Generator) -> pd.DataFrame:
    feat = cfg["feature_names"]
    rows = []

    # 0〜1 / 0〜100 両対応（内部は 0〜100 前提）
    def read_percent_pair(vmin, vmax, fallback=(0.0, 100.0)):
        if vmin is None and vmax is None:
            vmin, vmax = fallback
        if vmin is None: vmin = vmax
        if vmax is None: vmax = vmin
        # 0〜1 の場合は 100 倍
        if max(vmin, vmax) <= 1.0:
            vmin, vmax = vmin * 100.0, vmax * 100.0
        return float(vmin), float(vmax)

    # ===== 入力（lower/upper）を内部キーへマップ =====
    # EnergyCategory (1..8)
    syoene = int(max(1, min(8, case_params.get("EnergyCategory_-", 1))))

    # BPI
    BPI_min = float(case_params.get(
        "lower_BPI_-", np.nan)) if "lower_BPI_-" in case_params else None
    BPI_max = float(case_params.get(
        "upper_BPI_-", np.nan)) if "upper_BPI_-" in case_params else None

    # 断熱厚さ
    dmin = int(
        case_params.get("lower_Insulation_mm",
                        50)) if "lower_Insulation_mm" in case_params else 50
    dmax = int(
        case_params.get("upper_Insulation_mm",
                        100)) if "upper_Insulation_mm" in case_params else 100

    # WWR（0〜1 指定 → 0〜100 に変換）
    kaiko_min_in = case_params.get("lower_WWR_-", None)
    kaiko_max_in = case_params.get("upper_WWR_-", None)
    kaiko_min, kaiko_max = read_percent_pair(kaiko_min_in, kaiko_max_in,
                                             (0.0, 100.0))

    # U値: 入力に無ければ広め
    U_min = float(case_params.get(
        "lower_U-value_-", 1.0)) if "lower_U-value_-" in case_params else 1.0
    U_max = float(case_params.get(
        "upper_U-value_-", 3.0)) if "upper_U-value_-" in case_params else 3.0

    # SHGC（0.20〜0.50を既定）
    nmin = float(case_params.get(
        "lower_SHGC_-", 0.20)) if "lower_SHGC_-" in case_params else 0.20
    nmax = float(case_params.get(
        "upper_SHGC_-", 0.50)) if "upper_SHGC_-" in case_params else 0.50

    # NS/EW/AreaRatio
    NS_min = case_params.get("lower_VerticalLength_mm", None)
    NS_max = case_params.get("upper_VerticalLength_mm", None)
    EW_min = case_params.get("lower_HorizontalLength_mm", None)
    EW_max = case_params.get("upper_HorizontalLength_mm", None)
    AR_min = case_params.get("lower_CoverageRatio_-", None)
    AR_max = case_params.get("upper_CoverageRatio_-", None)

    # 壁面3変数の生成（新方式）
    NS_all, EW_all, AR_all = _sample_walls_case_aware(
        n=n_random,
        rng=rng,
        ns_range=(None if NS_min is None else float(NS_min),
                  None if NS_max is None else float(NS_max)),
        ew_range=(None if EW_min is None else float(EW_min),
                  None if EW_max is None else float(EW_max)),
        ar_range=(None if AR_min is None else float(AR_min),
                  None if AR_max is None else float(AR_max)),
        seed=42)

    step = float(cfg.get("grid_step", 0.01))

    # 設備（0/1 ランダム）
    setsubi_all = rng.integers(0, 2, size=n_random, endpoint=False)

    # 連続値（グリッドから抽選）
    dannetu_all = _int_choice(dmin, dmax, n_random, rng)
    kaikouritu_all = _grid_choice(kaiko_min, kaiko_max, step, n_random,
                                  rng)  # 0-100（内部保持）
    uchi_all = _grid_choice(U_min, U_max, step, n_random, rng)
    nissya_all = _grid_choice(nmin, nmax, step, n_random, rng)

    # 行を構築
    for i in range(n_random):
        rec = {}
        for j, name in enumerate(feat["syoene_onehot"], start=1):
            rec[name] = int(j == syoene)
        rec["setsubi_1"] = int(setsubi_all[i])
        rec["setsubi_0"] = int(1 - setsubi_all[i])

        rec["NS_Wall"] = float(NS_all[i])
        rec["EW_Wall"] = float(EW_all[i])
        rec["AreaRatio"] = float(AR_all[i])
        rec["kaikouritu"] = float(kaikouritu_all[i])  # 0-100
        rec["U-chi"] = float(uchi_all[i])
        rec["nissyasyutoku"] = float(nissya_all[i])  # 0-1
        rec["dannetu"] = int(dannetu_all[i])
        rows.append(rec)

    df = pd.DataFrame(rows).reindex(columns=feature_order)

    # 型揃え
    for col in [
            "NS_Wall", "EW_Wall", "AreaRatio", "dannetu", "kaikouritu",
            "U-chi", "nissyasyutoku"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for c in [
            c for c in df.columns
            if c.startswith("syoene_") or c.startswith("setsubi_")
    ]:
        df[c] = df[c].astype(int)

    if df.isnull().any().any():
        missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Generated candidates contain NaN columns: {missing}")

    return df


# ========== BPI 条件 ==========
def build_bpi_filter(case_params: dict, cfg: dict):
    if "lower_BPI_-" in case_params or "upper_BPI_-" in case_params:
        lo = float(case_params.get("lower_BPI_-", -np.inf))
        hi = float(case_params.get("upper_BPI_-", np.inf))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    tgt = float(case_params.get("BPI", 0.85))
    tol = float(case_params.get("tol", cfg.get("bpi_tolerance", 0.02)))
    return tgt - tol, tgt + tol


# ========== 代表8点（重複禁止 & 代替なしなら欠番はNaN行） ==========
SELECT_NAMES_EN = [
    "InternalHeat_1",
    "InternalHeat_0",
    "U-value_Max",
    "U-value_Min",
    "SHGC_Max",
    "SHGC_Min",
    "Insulation_Max",
    "Insulation_Min",
]


def _take_first_unique(sorted_index: pd.Index, chosen: set[int]) -> int | None:
    for i in list(sorted_index):
        i = int(i)
        if i not in chosen:
            return i
    return None


def pick_representatives(df: pd.DataFrame) -> list[int | None]:
    chosen: set[int] = set()
    sel: list[int | None] = []

    # ① InternalHeat=1
    idx = _take_first_unique(df.index[df["setsubi_1"] == 1], chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ② InternalHeat=0
    idx = _take_first_unique(df.index[df["setsubi_0"] == 1], chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ③ U値 最大
    idx = _take_first_unique(
        df.sort_values("U-chi", ascending=False).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ④ U値 最小
    idx = _take_first_unique(
        df.sort_values("U-chi", ascending=True).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ⑤ SHGC 最大
    idx = _take_first_unique(
        df.sort_values("nissyasyutoku", ascending=False).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ⑥ SHGC 最小
    idx = _take_first_unique(
        df.sort_values("nissyasyutoku", ascending=True).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ⑦ 断熱厚さ 最大
    idx = _take_first_unique(
        df.sort_values("dannetu", ascending=False).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ⑧ 断熱厚さ 最小
    idx = _take_first_unique(
        df.sort_values("dannetu", ascending=True).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    return sel


# ========== 出力ブロック生成 ==========
REP_COLS = [
    "CaseName", "EnergyCategory_-", "InternalHeat_-", "VerticalLength_mm",
    "HorizontalLength_mm", "CoverageRatio_-", "Insulation_mm", "WWR_-",
    "U-value_-", "SHGC_-", "predictBEI_-", "predictBPI_-",
    "predictCooling_MJm2", "predictHeating_MJm2"
]


def build_summary_row(case_name: str, params: dict) -> pd.DataFrame:
    """ブロック1：入力レンジ 1行"""
    cols = [
        "CaseName",
        "EnergyCategory_-",
        "lower_BPI_-",
        "upper_BPI_-",
        "lower_Insulation_mm",
        "upper_Insulation_mm",
        "lower_WWR_-",
        "upper_WWR_-",
        "lower_VerticalLength_mm",
        "upper_VerticalLength_mm",
        "lower_HorizontalLength_mm",
        "upper_HorizontalLength_mm",
        "lower_CoverageRatio_-",
        "upper_CoverageRatio_-",
    ]

    row = {
        "CaseName":
        case_name,
        "EnergyCategory_-":
        int(params.get("EnergyCategory_-", 1)),
        "lower_BPI_-":
        params.get("lower_BPI_-", np.nan),
        "upper_BPI_-":
        params.get("upper_BPI_-", np.nan),
        "lower_Insulation_mm":
        params.get("lower_Insulation_mm", np.nan),
        "upper_Insulation_mm":
        params.get("upper_Insulation_mm", np.nan),
        "lower_WWR_-":
        params.get("lower_WWR_-", np.nan),
        "upper_WWR_-":
        params.get("upper_WWR_-", np.nan),
        "lower_VerticalLength_mm":
        params.get("lower_VerticalLength_mm", np.nan),
        "upper_VerticalLength_mm":
        params.get("upper_VerticalLength_mm", np.nan),
        "lower_HorizontalLength_mm":
        params.get("lower_HorizontalLength_mm", np.nan),
        "upper_HorizontalLength_mm":
        params.get("upper_HorizontalLength_mm", np.nan),
        "lower_CoverageRatio_-":
        params.get("lower_CoverageRatio_-", np.nan),
        "upper_CoverageRatio_-":
        params.get("upper_CoverageRatio_-", np.nan),
    }
    df = pd.DataFrame([row], columns=cols)
    # 数値丸め
    for c in df.columns:
        if c in ("CaseName", ): continue
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


def build_representative_block(df_rep_src: pd.DataFrame, preds: np.ndarray,
                               syoene_value: int) -> pd.DataFrame:
    """ブロック2：代表8行（存在する分だけ）"""
    # 出力用に整形（2桁丸め、WWRは 0-100 → 0-1 に戻す）
    out = pd.DataFrame({
        "CaseName":
        df_rep_src["_selector_name"].astype(str),
        "EnergyCategory_-":
        int(syoene_value),
        "InternalHeat_-":
        pd.to_numeric(df_rep_src["setsubi_1"],
                      errors="coerce").round(0).astype("Int64"),
        "VerticalLength_mm":
        pd.to_numeric(df_rep_src["NS_Wall"], errors="coerce").round(2),
        "HorizontalLength_mm":
        pd.to_numeric(df_rep_src["EW_Wall"], errors="coerce").round(2),
        "CoverageRatio_-":
        pd.to_numeric(df_rep_src["AreaRatio"], errors="coerce").round(2),
        "Insulation_mm":
        pd.to_numeric(df_rep_src["dannetu"],
                      errors="coerce").round(0).astype("Int64"),
        "WWR_-": (pd.to_numeric(df_rep_src["kaikouritu"], errors="coerce") /
                  100.0).round(2),
        "U-value_-":
        pd.to_numeric(df_rep_src["U-chi"], errors="coerce").round(2),
        "SHGC_-":
        pd.to_numeric(df_rep_src["nissyasyutoku"], errors="coerce").round(2),
    })

    pred_df = pd.DataFrame(preds, columns=["Cooling", "Heating", "BEI", "BPI"])
    out["predictBEI_-"] = pd.to_numeric(pred_df["BEI"],
                                        errors="coerce").round(2)
    out["predictBPI_-"] = pd.to_numeric(pred_df["BPI"],
                                        errors="coerce").round(2)
    out["predictCooling_MJm2"] = pd.to_numeric(pred_df["Cooling"],
                                               errors="coerce").round(2)
    out["predictHeating_MJm2"] = pd.to_numeric(pred_df["Heating"],
                                               errors="coerce").round(2)

    return out.reindex(columns=REP_COLS)


# ========== メイン ==========
def main():
    parser = argparse.ArgumentParser(
        description="Pref2Cond: 性能→条件探索（ケース別CSV出力：サマリ + 代表8行）")
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Pref2Cond.csv（縦持ち）省略時は実行ファイルと同じフォルダ")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="model_1019.cbm（省略時は同梱）")
    parser.add_argument("--config",
                        type=str,
                        default=None,
                        help="設定YAML（省略時は同梱）")
    parser.add_argument("--outdir",
                        type=str,
                        default=None,
                        help="出力フォルダ（既定：実行ファイルと同じフォルダ）")
    args = parser.parse_args()

    base = executable_dir()
    in_path = args.input or os.path.join(base, "Pref2Cond.csv")
    out_dir = args.outdir or base
    cfg_path = args.config or resource_path(
        os.path.join("app_resources", "pref2cond_config.yaml"))
    model_path = args.model or resource_path(
        os.path.join("app_resources", "model_1019.cbm"))

    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(load_yaml_if_exists(cfg_path))

    # モデル
    model = load_model(model_path)
    feature_order = list(model.feature_names_)
    model_out_order = cfg.get("model_output_order",
                              DEFAULT_CONFIG["model_output_order"])
    if model_out_order != ["Cooling", "Heating", "BEI", "BPI"]:
        warnings.warn("model_output_order が既定と異なります。出力カラム順が想定とズレる可能性があります。")

    # 入力CSV → ケース辞書
    df_in = read_pref2cond_csv(in_path)
    cases = parse_cases(df_in)

    rng = np.random.default_rng(42)
    os.makedirs(out_dir, exist_ok=True)

    for case_key, params in cases.items():
        # サマリ行の CaseName（入力に CaseName がなければ列名を使う）
        case_name_for_summary = str(params.get("CaseName", case_key))

        # 候補生成
        n_random = int(cfg.get("n_random", 20000))
        X = make_candidates(params, cfg, feature_order, n_random, rng)

        # 予測
        pred = np.asarray(model.predict(X))
        if pred.ndim != 2 or pred.shape[1] != len(model_out_order):
            raise ValueError(
                "Model must output 4 targets (Cooling, Heating, BEI, BPI) with consistent order."
            )

        # BPI フィルタ
        bpi_idx = model_out_order.index("BPI")
        bpi_min, bpi_max = build_bpi_filter(params, cfg)
        mask = (pred[:, bpi_idx] >= bpi_min) & (pred[:, bpi_idx] <= bpi_max)
        X_good = X.loc[mask].copy()
        pred_good = pred[mask, :]

        # 代表8点（重複禁止）
        X_good["_selector_name"] = ""
        idxs = pick_representatives(X_good)

        rep_df_rows = []
        rep_pred_rows = []
        for nm, idx in zip(SELECT_NAMES_EN, idxs):
            if idx is None:
                continue
            r = X_good.loc[int(idx)].copy()
            r["_selector_name"] = nm
            rep_df_rows.append(r)
            rep_pred_rows.append(pred_good[X_good.index.get_loc(
                int(idx))].tolist())

        # 出力ファイル
        safe_case = str(case_key).replace("/", "_")
        out_path = os.path.join(out_dir, f"Pref2Cond_result_{safe_case}.csv")

        # ブロック1（サマリ）
        summary_df = build_summary_row(case_name_for_summary, params)

        # 書き出し
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            summary_df.to_csv(f, index=False)
            f.write("\n")  # 空行

            if len(rep_df_rows) == 0:
                # 結果なし
                f.write("- No results found. -\n")
            else:
                rep_src = pd.DataFrame(rep_df_rows).reset_index(drop=True)
                rep_block = build_representative_block(
                    rep_src,
                    np.array(rep_pred_rows, dtype=float),
                    syoene_value=int(params.get("EnergyCategory_-", 1)))
                rep_block.to_csv(f, index=False)

        print(f"[{case_key}] -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
