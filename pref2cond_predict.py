# -*- coding: utf-8 -*-
"""
pref2cond_predict.py  (要望反映版)
- 入力: Pref2Cond.csv（縦持ち: 1列目=パラメータ名, 2列目以降=case列）
- 出力: 各caseごとに 8行のCSV（設備=1/0, U値max/min, 日射取得率max/min, 断熱厚さmax/min）
- モデル: model.cbm（CatBoost MultiRMSE: ["Cooling","Heating","BEI","BPI"]）
- 要件:
  1) 8行固定。代表が見つからない項目は「全て NaN」
  2) 小数は全て小数第2位で丸め（予測値, U値, 日射取得率, 壁面長さ, 面積率）
  3) 代表サンプルは重複禁止（同一indexを再利用しない）
  4) 整数列: 省エネ区分/設備/断熱厚さ_mm は Int64（nullable）
  5) 開口率_% は従来どおり整数（%）
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
    "NS_Wall": {
        "choices": [136.4616, 180.7740, 197.6436, 204.6924, 271.1808]
    },
    "EW_Wall": {
        "choices": [204.6924, 271.1808, 131.7492, 136.4616, 180.7740]
    },
    "AreaRatio": {
        "choices": [1.000000, 0.932252, 0.531220]
    },
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
    ns_min, ns_max = ns_range
    ew_min, ew_max = ew_range
    ar_min, ar_max = ar_range

    any_constraint = any(
        v is not None
        for v in [ns_min, ns_max, ew_min, ew_max, ar_min, ar_max])
    pats = (PATTERNS[_mask_patterns_by_range(PATTERNS, ns_min, ns_max, ew_min,
                                             ew_max, ar_min, ar_max)].copy()
            if any_constraint else PATTERNS.copy())
    # ★ ここがポイント：位置インデックスに揃える
    pats = pats.reset_index(drop=True)

    if len(pats) > 0:
        alloc = _alloc_counts_to_total(pats, total=n, seed=seed)
        NS = np.empty(n, dtype=float)
        EW = np.empty(n, dtype=float)
        AR = np.empty(n, dtype=float)
        pos = 0
        # ★ k は 0..len(pats)-1 の位置インデックスで回す
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

    # ③ 該当0件：範囲から一様（件数配分は PATTERNS 比率）
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
    # こちらは PATTERNS 自体が 0.. の連番なのでそのままでOK
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


# ========== Pref2Cond.csv の読み取り ==========
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
    key_map = {
        "省エネ区分": "syoene",
        "syoene": "syoene",
        "BPI": "BPI",
        "BPI_target": "BPI",
        "BPI_tgt": "BPI",
        "BPImin": "BPI_min",
        "BPI_min": "BPI_min",
        "BPImax": "BPI_max",
        "BPI_max": "BPI_max",
        "tol": "tol",
        "TOLERANCE": "tol",
        "開口率_min": "kaiko_min",
        "kaikouritu_min": "kaiko_min",
        "開口率_max": "kaiko_max",
        "kaikouritu_max": "kaiko_max",
        "断熱厚さ_min": "dannetu_min",
        "dannetu_min": "dannetu_min",
        "断熱厚さ_max": "dannetu_max",
        "dannetu_max": "dannetu_max",
        "U値_min": "U_min",
        "U_min": "U_min",
        "U値_max": "U_max",
        "U_max": "U_max",
        "日射取得率_min": "nissya_min",
        "nissyasyutoku_min": "nissya_min",
        "日射取得率_max": "nissya_max",
        "nissyasyutoku_max": "nissya_max",
        "NS_Wall_min": "NS_min",
        "NSmin": "NS_min",
        "NS_Wall_max": "NS_max",
        "NSmax": "NS_max",
        "EW_Wall_min": "EW_min",
        "EWmin": "EW_min",
        "EW_Wall_max": "EW_max",
        "EWmax": "EW_max",
        "AreaRatio_min": "AR_min",
        "AreaRatiomin": "AR_min",
        "AreaRatio_max": "AR_max",
        "AreaRatiomax": "AR_max",
        "NS_Wall": "NS_fix",
        "EW_Wall": "EW_fix",
        "AreaRatio": "AR_fix",
    }
    first = df.columns[0]
    cases = {}
    for c in df.columns[1:]:
        params = {}
        for _, row in df.iterrows():
            raw_key = str(row[first]).strip()
            key = key_map.get(raw_key, raw_key)
            val = row[c]
            if pd.isna(val):
                continue
            try:
                val_num = float(str(val).strip())
                params[key] = val_num
            except Exception:
                params[key] = str(val).strip()
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

    # %は 0-1 / 0-100 両対応（内部は 0-100 前提）
    def read_percent_pair(min_key, max_key, fallback):
        vmin = case_params.get(min_key, None)
        vmax = case_params.get(max_key, None)
        if vmin is None and vmax is None:
            vmin, vmax = fallback
        if max(vmin, vmax) <= 1.0:
            vmin, vmax = vmin * 100.0, vmax * 100.0
        return vmin, vmax

    kaiko_min, kaiko_max = read_percent_pair("kaiko_min", "kaiko_max",
                                             (0.0, 100.0))
    U_min = case_params.get("U_min", 1.0)
    U_max = case_params.get("U_max", 3.0)
    nmin = case_params.get("nissya_min", 0.20)
    nmax = case_params.get("nissya_max", 0.50)
    dmin = int(case_params.get("dannetu_min", 50))
    dmax = int(case_params.get("dannetu_max", 100))
    step = float(cfg.get("grid_step", 0.01))

    # --- ★ここから壁面3変数の生成ロジックを新方式に ---
    # 固定が指定されていたら、範囲として (val,val) を使う（パターン一致があれば比率配分）
    def _read_fix_or_range(fix_key, min_key, max_key):
        if fix_key in case_params:
            v = float(case_params[fix_key])
            return (v, v)
        vmin = case_params.get(min_key, None)
        vmax = case_params.get(max_key, None)
        return (None if vmin is None else float(vmin),
                None if vmax is None else float(vmax))

    ns_range = _read_fix_or_range("NS_fix", "NS_min", "NS_max")
    ew_range = _read_fix_or_range("EW_fix", "EW_min", "EW_max")
    ar_range = _read_fix_or_range("AR_fix", "AR_min", "AR_max")

    NS_all, EW_all, AR_all = _sample_walls_case_aware(n=n_random,
                                                      rng=rng,
                                                      ns_range=ns_range,
                                                      ew_range=ew_range,
                                                      ar_range=ar_range,
                                                      seed=42)
    # --- ★ここまで差し替え ---

    # そのほかの列は従来どおりバランス良く
    # syoene（1..8）
    syoene = int(max(1, min(8, case_params.get("syoene", 1))))
    syoene_all = np.full(n_random, syoene, dtype=int)

    # 設備（0/1 ランダム）
    setsubi_all = rng.integers(0, 2, size=n_random, endpoint=False)

    # 連続値（グリッドから抽選）
    def _grid_choice(low: float, high: float, step: float,
                     size: int) -> np.ndarray:
        grid = np.arange(float(low), float(high) + 1e-12, float(step))
        if len(grid) == 0:
            grid = np.array([float(low)])
        return rng.choice(grid, size=size, replace=True)

    dannetu_all = rng.integers(dmin, dmax + 1, size=n_random, endpoint=True)
    kaikouritu_all = _grid_choice(kaiko_min, kaiko_max, step,
                                  n_random)  # 0-100想定（%）
    uchi_all = _grid_choice(U_min, U_max, step, n_random)
    nissya_all = _grid_choice(nmin, nmax, step, n_random)

    # 行を構築
    for i in range(n_random):
        rec = {}
        for j, name in enumerate(feat["syoene_onehot"], start=1):
            rec[name] = (j == syoene_all[i])
        rec["setsubi_1"] = int(setsubi_all[i])
        rec["setsubi_0"] = int(1 - setsubi_all[i])

        rec["NS_Wall"] = float(NS_all[i])
        rec["EW_Wall"] = float(EW_all[i])
        rec["AreaRatio"] = float(AR_all[i])
        rec["kaikouritu"] = float(kaikouritu_all[i])
        rec["U-chi"] = float(uchi_all[i])
        rec["nissyasyutoku"] = float(nissya_all[i])
        rec["dannetu"] = int(dannetu_all[i])
        rows.append(rec)

    df = pd.DataFrame(rows).reindex(columns=feature_order)

    # 型揃え（学習時と一致：数値= float、カテゴリ= Int）
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
    if "BPI_min" in case_params or "BPI_max" in case_params:
        lo = float(case_params.get("BPI_min", -np.inf))
        hi = float(case_params.get("BPI_max", np.inf))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    tgt = float(case_params.get("BPI", 0.85))
    tol = float(case_params.get("tol", cfg.get("bpi_tolerance", 0.02)))
    return tgt - tol, tgt + tol


# ========== 代表8点（重複禁止 & 代替なしなら欠番はNaN行） ==========
SELECT_NAMES = [
    "設備発熱=1",
    "設備発熱=0",
    "U値_最大",
    "U値_最小",
    "日射取得率_最大",
    "日射取得率_最小",
    "断熱厚さ_最大",
    "断熱厚さ_最小",
]


def _take_first_unique(sorted_index: pd.Index, chosen: set[int]) -> int | None:
    for i in list(sorted_index):
        i = int(i)
        if i not in chosen:
            return i
    return None


def pick_representatives(df: pd.DataFrame) -> list[int | None]:
    """
    8つのカテゴリに対して、インデックス重複なしで選ぶ。
    見つからないカテゴリは None を返す（後で NaN 行にする）。
    """
    chosen: set[int] = set()
    sel: list[int | None] = []

    # ① setsubi_1=True
    idx = _take_first_unique(df.index[df["setsubi_1"] == 1], chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ② setsubi_0=True
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

    # ⑤ 日射取得率 最大
    idx = _take_first_unique(
        df.sort_values("nissyasyutoku", ascending=False).index, chosen)
    if idx is not None: chosen.add(idx)
    sel.append(idx)

    # ⑥ 日射取得率 最小
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


# ========== 出力整形 ==========
FINAL_COLS = [
    "条件名", "省エネ区分", "設備", "平面_縦長さ", "平面_横長さ", "平面_面積率", "断熱厚さ_mm", "開口率_%",
    "U値", "日射取得率", "予測BEI", "予測BPI", "予測Cooling", "予測Heating"
]


def _syoene_series(df_rep: pd.DataFrame) -> pd.Series:
    cols = [c for c in df_rep.columns if c.startswith("syoene_")]
    if not cols:
        return pd.Series([pd.NA] * len(df_rep), dtype="Int64")
    arr = df_rep[cols].fillna(0).astype(int).to_numpy()
    # 全行で all zero → 省エネ区分は NaN
    row_sums = arr.sum(axis=1)
    idx = np.where(row_sums > 0, np.argmax(arr, axis=1) + 1, -1)
    s = pd.Series(idx, dtype="Int64")
    s = s.mask(s < 0, other=pd.NA)
    return s


def _round2(s: pd.Series) -> pd.Series:
    # すべて 2桁に揃える（数値のみ）
    s = pd.to_numeric(s, errors="coerce")
    return s.round(2)


def to_output_rows(df_rep: pd.DataFrame, y_pred: np.ndarray,
                   model_out_order: list[str]) -> pd.DataFrame:
    # 予測配列（(n, 4)）→ DataFrame、2桁
    pred_df = pd.DataFrame(y_pred, columns=model_out_order)
    for c in pred_df.columns:
        pred_df[c] = _round2(pred_df[c])

    # 省エネ区分
    syo_series = _syoene_series(df_rep)

    # 出力本体
    out = pd.DataFrame({
        "条件名":
        df_rep["_selector_name"].fillna("").astype("object"),
        # 条件不一致の行は 省エネ区分も含めて全 NaN（syo_series で担保）
        "省エネ区分":
        syo_series,
        "設備":
        pd.to_numeric(df_rep["setsubi_1"],
                      errors="coerce").round(0).astype("Int64"),
        # 小数 2 桁で丸め（壁面長さ/面積率も2桁）
        "平面_縦長さ":
        _round2(df_rep["NS_Wall"]),
        "平面_横長さ":
        _round2(df_rep["EW_Wall"]),
        "平面_面積率":
        _round2(df_rep["AreaRatio"]),
        # 整数列：断熱厚さ_mm, 開口率_%（%は整数運用継続）
        "断熱厚さ_mm":
        pd.to_numeric(df_rep["dannetu"],
                      errors="coerce").round(0).astype("Int64"),
        "開口率_%":
        pd.to_numeric(df_rep["kaikouritu"],
                      errors="coerce").round(0).astype("Int64"),
        # 小数2桁
        "U値":
        _round2(df_rep["U-chi"]),
        "日射取得率":
        _round2(df_rep["nissyasyutoku"]),
        "予測BEI":
        pred_df.get("BEI"),
        "予測BPI":
        pred_df.get("BPI"),
        "予測Cooling":
        pred_df.get("Cooling"),
        "予測Heating":
        pred_df.get("Heating"),
    })

    # 「全NaNにする」— 判定: 省エネ区分/設備/U値 などがすべて NaN かどうかで行毎に判断
    # ただし _selector_name は表示名なので除外
    num_cols = [c for c in FINAL_COLS if c not in ("条件名", )]
    na_mask = out[num_cols].isna().all(axis=1)
    # 既に NaN ならそのまま。ここでは追加処理不要（syo_series/各列が NaN）

    return out.reindex(columns=FINAL_COLS)


# ========== メイン ==========
def main():
    parser = argparse.ArgumentParser(
        description="Pref2Cond: 性能→条件探索（ケース別CSV出力）")
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Pref2Cond.csv（縦持ち）省略時は実行ファイルと同じフォルダ")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="model.cbm（省略時は同梱）")
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
        os.path.join("app_resources", "model.cbm"))

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

    # 入力CSV → ケース辞書
    df_in = read_pref2cond_csv(in_path)
    cases = parse_cases(df_in)

    rng = np.random.default_rng(42)
    os.makedirs(out_dir, exist_ok=True)

    made_files = []
    for case_name, params in cases.items():
        n_random = int(cfg.get("n_random", 20000))
        X = make_candidates(params, cfg, feature_order, n_random, rng)

        # 予測
        raw_pred = model.predict(X)
        pred = np.asarray(raw_pred)
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
        X_good["_selector_name"] = ""  # 後で埋める
        idxs = pick_representatives(X_good)

        # 代表 DF と予測配列（None → NaN行）
        rep_df_rows = []
        rep_pred_rows = []
        for nm, idx in zip(SELECT_NAMES, idxs):
            if idx is None:
                # 欠番 → 全て NaN の object 行（_selector_name のみ説明用に保持）
                empty = pd.Series({col: np.nan
                                   for col in X_good.columns},
                                  dtype="object")
                empty["_selector_name"] = nm
                rep_df_rows.append(empty)
                rep_pred_rows.append([np.nan, np.nan, np.nan, np.nan])
            else:
                r = X_good.loc[int(idx)].copy().astype("object")
                r["_selector_name"] = nm
                rep_df_rows.append(r)
                rep_pred_rows.append(pred_good[X_good.index.get_loc(
                    int(idx))].tolist())

        rep_df = pd.DataFrame(rep_df_rows).reset_index(drop=True)
        rep_df["_selector_name"] = rep_df["_selector_name"].astype("object")
        rep_pred = np.array(rep_pred_rows, dtype=float)

        out_df = to_output_rows(rep_df, rep_pred, model_out_order)

        # 保存
        safe_case = str(case_name).replace("/", "_")
        out_path = os.path.join(out_dir, f"Pref2Cond_result_{safe_case}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        made_files.append(out_path)
        print(f"[{case_name}] {len(out_df)} rows -> {out_path}")

    print("Done. Files:")
    for p in made_files:
        print(" -", p)


if __name__ == "__main__":
    main()
