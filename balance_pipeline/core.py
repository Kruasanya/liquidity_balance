from __future__ import annotations

import json
import math
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PipelineConfig:
    main_file: str = "Project 1_2024.xlsx"
    macro_file: str = "Инфляция и ключевая ставка Банка России_F01_01_2017_T29_05_2026.xlsx"
    ruonia_file: str = "RC_F01_01_2017_T28_05_2026.xlsx"
    usd_file: str = "RC_F01_01_2017_T30_05_2026.xlsx"
    forecast_date: Optional[str] = None
    artifacts_dir: str = "artifacts"
    error_threshold: float = 0.42
    rolling_window: int = 500
    cv_splits: int = 5
    cv_validation_size: int = 40
    min_train_size: int = 180
    random_state: int = 42
    allow_external_ffill: bool = True
    external_max_staleness_days: int = 7
    drift_window: int = 20


LAGS = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20]
TARGET_COL = "target_balance"
REQUIRED_MAIN_COLUMNS = ["Date", "Income", "Outcome", "Balance"]


def resolve_forecast_date(main_df: pd.DataFrame, forecast_date: Optional[str]) -> pd.Timestamp:
    if forecast_date:
        return pd.to_datetime(forecast_date).normalize()
    last_date = pd.to_datetime(main_df["Date"]).max().normalize()
    return (last_date + pd.offsets.BDay(1)).normalize()


def read_main_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data")
    missing = set(REQUIRED_MAIN_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В основном файле нет обязательных колонок: {sorted(missing)}")
    df = df[REQUIRED_MAIN_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for col in ["Income", "Outcome", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def _parse_month_dot_year(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if "." in text and len(text.split(".")) == 2:
        month, year = text.split(".")
        return pd.Timestamp(int(year), int(month), 1)
    return pd.to_datetime(value).to_period("M").to_timestamp()


def read_external_data(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro = pd.read_excel(config.macro_file)
        ruonia = pd.read_excel(config.ruonia_file, sheet_name="RC")
        usd = pd.read_excel(config.usd_file, sheet_name="RC")

    macro = macro.rename(
        columns={
            "Дата": "Date",
            "Ключевая ставка, % годовых": "key_rate",
            "Инфляция, % г/г": "inflation_yoy",
            "Цель по инфляции": "inflation_target",
        }
    )
    macro = macro[["Date", "key_rate", "inflation_yoy", "inflation_target"]].copy()
    macro["Date"] = macro["Date"].apply(_parse_month_dot_year)
    macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce").dt.normalize()
    macro = macro.sort_values("Date").drop_duplicates("Date")
    macro["key_rate_change_1m"] = macro["key_rate"].diff()
    macro["inflation_yoy_lag_1m"] = macro["inflation_yoy"].shift(1)
    macro["inflation_gap_lag_1m"] = (
        macro["inflation_yoy"].shift(1) - macro["inflation_target"].shift(1)
    )

    ruonia = ruonia.rename(columns={"DT": "Date", "ruo": "ruonia", "vol": "ruonia_volume"})
    ruonia = ruonia[["Date", "ruonia", "ruonia_volume"]].copy()
    ruonia["Date"] = pd.to_datetime(ruonia["Date"], errors="coerce").dt.normalize()
    ruonia = ruonia.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")

    usd = usd.rename(columns={"data": "Date", "curs": "usd_rub"})
    usd = usd[["Date", "usd_rub"]].copy()
    usd["Date"] = pd.to_datetime(usd["Date"], errors="coerce").dt.normalize()
    usd = usd.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")

    return {"macro": macro, "ruonia": ruonia, "usd": usd}


def validate_main_data(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    checks.append(("required_columns", "ok", "Все обязательные колонки есть"))
    null_dates = int(df["Date"].isna().sum())
    checks.append(("date_parse", "critical" if null_dates else "ok", f"Некорректных дат: {null_dates}"))
    duplicate_dates = int(df["Date"].duplicated().sum())
    checks.append(
        ("duplicate_dates", "critical" if duplicate_dates else "ok", f"Дубликатов дат: {duplicate_dates}")
    )
    missing_values = int(df[REQUIRED_MAIN_COLUMNS].isna().sum().sum())
    checks.append(("missing_values", "critical" if missing_values else "ok", f"Пропусков: {missing_values}"))
    is_sorted = df["Date"].is_monotonic_increasing
    checks.append(("time_sort", "ok" if is_sorted else "warning", f"Отсортировано по времени: {is_sorted}"))

    if df["Date"].notna().all():
        full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        missing_dates = len(set(full_range) - set(df["Date"]))
    else:
        missing_dates = None
    checks.append(
        (
            "calendar_continuity",
            "warning" if missing_dates else "ok",
            f"Пропущенных календарных дат: {missing_dates}",
        )
    )
    balance_gap = (df["Balance"] - (df["Income"] - df["Outcome"])).abs()
    large_gap = int((balance_gap > 1e-3).sum())
    checks.append(
        (
            "balance_income_outcome_gap",
            "warning" if large_gap else "ok",
            f"Строк с Balance != Income - Outcome при tol=1e-3: {large_gap}",
        )
    )
    report = pd.DataFrame(checks, columns=["check", "status", "details"])
    critical = report.loc[report["status"].eq("critical")]
    if not critical.empty:
        raise ValueError("Критические ошибки качества данных:\n" + critical.to_string(index=False))
    return report


def _tax_day_for_month(year: int, month: int) -> pd.Timestamp:
    candidate = pd.Timestamp(year, month, 28)
    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate.normalize()


def add_calendar_and_tax_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["is_business_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_business_month_end"] = out["Date"].dt.is_month_end.astype(int)
    out["is_business_quarter_end"] = out["Date"].dt.is_quarter_end.astype(int)
    tax_days = out["Date"].apply(lambda d: _tax_day_for_month(d.year, d.month))
    out["is_tax_day_28"] = (out["Date"] == tax_days).astype(int)
    out["is_business_day_before_tax_28"] = (out["Date"] == (tax_days - pd.offsets.BDay(1)).dt.normalize()).astype(int)
    out["is_business_day_after_tax_28"] = (out["Date"] == (tax_days + pd.offsets.BDay(1)).dt.normalize()).astype(int)
    out["is_tax_window_28"] = (
        out[["is_tax_day_28", "is_business_day_before_tax_28", "is_business_day_after_tax_28"]].max(axis=1)
    )
    return out


def _make_daily_external_frame(
    date_index: pd.Series, external: Dict[str, pd.DataFrame], config: PipelineConfig
) -> Tuple[pd.DataFrame, str]:
    daily = pd.DataFrame({"Date": pd.to_datetime(date_index).sort_values().unique()})
    daily = daily.sort_values("Date")

    macro = external["macro"].copy()
    daily["month_date"] = daily["Date"].dt.to_period("M").dt.to_timestamp()
    daily = daily.merge(macro.drop(columns=["inflation_yoy"]), left_on="month_date", right_on="Date", how="left", suffixes=("", "_macro"))
    daily = daily.drop(columns=["Date_macro", "month_date"], errors="ignore")

    for source_name, source_df, cols in [
        ("ruonia", external["ruonia"], ["ruonia", "ruonia_volume"]),
        ("usd", external["usd"], ["usd_rub"]),
    ]:
        source_df = source_df[["Date"] + cols].sort_values("Date").copy()
        daily = pd.merge_asof(daily.sort_values("Date"), source_df, on="Date", direction="backward")
        last_known = pd.merge_asof(
            daily[["Date"]].sort_values("Date"),
            source_df[["Date"]].rename(columns={"Date": f"{source_name}_known_date"}).sort_values(f"{source_name}_known_date"),
            left_on="Date",
            right_on=f"{source_name}_known_date",
            direction="backward",
        )
        daily[f"{source_name}_staleness_days"] = (
            daily["Date"] - last_known[f"{source_name}_known_date"]
        ).dt.days

    quality_bits = []
    external_cols = [
        "key_rate",
        "key_rate_change_1m",
        "inflation_yoy_lag_1m",
        "inflation_target",
        "inflation_gap_lag_1m",
        "ruonia",
        "ruonia_volume",
        "usd_rub",
    ]
    missing = {col: int(daily[col].isna().sum()) for col in external_cols if col in daily}
    if any(v > 0 for v in missing.values()):
        quality_bits.append(f"external_missing={missing}")
    stale_cols = [c for c in daily.columns if c.endswith("_staleness_days")]
    for c in stale_cols:
        max_stale = daily[c].max()
        if pd.notna(max_stale) and max_stale > config.external_max_staleness_days:
            quality_bits.append(f"{c}_max={int(max_stale)}")
    status = "ok" if not quality_bits else "warning: " + "; ".join(quality_bits)
    return daily, status


def compute_outlier_thresholds(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for col in ["Balance", "Income", "Outcome", TARGET_COL]:
        if col not in train_df:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        thresholds[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(q1 - 1.5 * iqr),
            "upper": float(q3 + 1.5 * iqr),
            "extreme_lower": float(series.quantile(0.01)),
            "extreme_upper": float(series.quantile(0.99)),
        }
    return thresholds


def add_lag_rolling_and_outlier_features(
    df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    out = df.copy()
    for source in ["Balance", "Income", "Outcome"]:
        base = source.lower()
        for lag in LAGS:
            out[f"{base}_lag_{lag}"] = out[source].shift(lag)
        windows = ROLLING_WINDOWS if source == "Balance" else [5, 10, 20]
        for window in windows:
            out[f"{base}_rolling_mean_{window}"] = out[source].shift(1).rolling(window).mean()
            if source == "Balance":
                out[f"{base}_rolling_std_{window}"] = out[source].shift(1).rolling(window).std()

    if "ruonia" in out:
        out["ruonia_lag_1"] = out["ruonia"].shift(1)
        out["ruonia_volume_lag_1"] = out["ruonia_volume"].shift(1)
        out["ruonia_spread_to_key_lag_1"] = (out["ruonia"] - out["key_rate"]).shift(1)
        out["ruonia_rolling_mean_5"] = out["ruonia"].shift(1).rolling(5).mean()
        out["ruonia_rolling_std_5"] = out["ruonia"].shift(1).rolling(5).std()
    if "usd_rub" in out:
        out["usd_rub_lag_1"] = out["usd_rub"].shift(1)
        out["usd_rub_return_1"] = out["usd_rub"].pct_change().shift(1)
        out["usd_rub_rolling_mean_5"] = out["usd_rub"].shift(1).rolling(5).mean()
        out["usd_rub_rolling_std_5"] = out["usd_rub"].shift(1).rolling(5).std()

    def is_outlier(series: pd.Series, col: str) -> pd.Series:
        if col not in thresholds:
            return pd.Series(False, index=series.index)
        low = thresholds[col]["lower"]
        high = thresholds[col]["upper"]
        return ((series < low) | (series > high)).fillna(False)

    out["balance_lag_1_is_iqr_outlier"] = is_outlier(out["balance_lag_1"], "Balance").astype(int)
    out["balance_lag_5_is_iqr_outlier"] = is_outlier(out["balance_lag_5"], "Balance").astype(int)
    out["income_lag_1_is_iqr_outlier"] = is_outlier(out["income_lag_1"], "Income").astype(int)
    out["outcome_lag_1_is_iqr_outlier"] = is_outlier(out["outcome_lag_1"], "Outcome").astype(int)
    for window in ROLLING_WINDOWS:
        col = f"balance_rolling_std_{window}"
        reference = out.loc[out["Date"] < out["Date"].max(), col].dropna()
        high = reference.quantile(0.90) if not reference.empty else np.nan
        out[f"{col}_is_high"] = (out[col] > high).fillna(False).astype(int)
    recent_cols = ["balance_lag_1_is_iqr_outlier", "balance_lag_5_is_iqr_outlier"]
    out["has_recent_balance_outlier"] = out[recent_cols].max(axis=1).astype(int)

    if TARGET_COL in out and TARGET_COL in thresholds:
        target = out[TARGET_COL]
        t = thresholds[TARGET_COL]
        out["target_balance_is_iqr_outlier"] = ((target < t["lower"]) | (target > t["upper"])).fillna(False).astype(int)
        out["target_balance_is_extreme_quantile"] = (
            (target < t["extreme_lower"]) | (target > t["extreme_upper"])
        ).fillna(False).astype(int)
        out["target_outlier_alert"] = (
            out["target_balance_is_iqr_outlier"].astype(bool)
            | out["target_balance_is_extreme_quantile"].astype(bool)
        ).astype(int)
    return out


def build_model_frame(
    main_df: pd.DataFrame,
    external: Dict[str, pd.DataFrame],
    config: PipelineConfig,
    forecast_date: pd.Timestamp,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, str]:
    facts = main_df.copy()
    facts = facts.sort_values("Date").drop_duplicates("Date")
    start = facts["Date"].min()
    end = max(facts["Date"].max(), forecast_date)
    calendar = pd.DataFrame({"Date": pd.date_range(start, end, freq="B")})
    frame = calendar.merge(facts, on="Date", how="left")
    frame[TARGET_COL] = frame["Balance"]
    frame = add_calendar_and_tax_features(frame)
    external_frame, external_status = _make_daily_external_frame(frame["Date"], external, config)
    frame = frame.merge(external_frame, on="Date", how="left")
    if thresholds is None:
        train_seed = frame.loc[frame["Date"] < forecast_date].copy()
        thresholds = compute_outlier_thresholds(train_seed)
    frame = add_lag_rolling_and_outlier_features(frame, thresholds)
    return frame, external_status


def get_feature_cols(frame: pd.DataFrame) -> List[str]:
    blocked = {
        "Date",
        "Income",
        "Outcome",
        "Balance",
        TARGET_COL,
        "target_balance_is_iqr_outlier",
        "target_balance_is_extreme_quantile",
        "target_outlier_alert",
        "ruonia",
        "ruonia_volume",
        "usd_rub",
        "ruonia_staleness_days",
        "usd_staleness_days",
    }
    cols = [c for c in frame.columns if c not in blocked]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
    return numeric


def calculate_model_profit(y_true: Iterable[float], y_pred: Iterable[float], key_rate: Iterable[float]) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    key_rate = np.asarray(key_rate, dtype=float) / 100
    derivative_rate = (key_rate + 0.005) / 365
    cb_overnight_rate = (key_rate - 0.009) / 365
    borrow_rate = (key_rate + 0.010) / 365
    placed_amount = np.maximum(y_pred, 0)
    residual_balance = y_true - placed_amount
    derivative_profit = placed_amount * derivative_rate
    residual_profit = np.where(
        residual_balance >= 0,
        residual_balance * cb_overnight_rate,
        residual_balance * borrow_rate,
    )
    return derivative_profit + residual_profit


def calculate_ideal_profit(y_true: Iterable[float], key_rate: Iterable[float]) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    key_rate = np.asarray(key_rate, dtype=float) / 100
    derivative_rate = (key_rate + 0.005) / 365
    borrow_rate = (key_rate + 0.010) / 365
    return np.where(y_true >= 0, y_true * derivative_rate, y_true * borrow_rate)


def evaluate_predictions(
    y_true: Iterable[float], y_pred: Iterable[float], key_rate: Iterable[float], threshold: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    key_rate = np.asarray(key_rate, dtype=float)
    errors = y_true - y_pred
    model_profit = calculate_model_profit(y_true, y_pred, key_rate)
    ideal_profit = calculate_ideal_profit(y_true, key_rate)
    business_loss = ideal_profit - model_profit
    return {
        "business_loss": float(np.nansum(business_loss)),
        "business_loss_mean": float(np.nanmean(business_loss)),
        "model_profit": float(np.nansum(model_profit)),
        "ideal_profit": float(np.nansum(ideal_profit)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "share_abs_error_le_0_42": float((np.abs(errors) <= threshold).mean()),
        "bias_mean_error": float(np.nanmean(errors)),
    }


def _make_estimator(name: str, config: PipelineConfig) -> Pipeline:
    if name == "Lasso":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=20000, random_state=config.random_state)),
            ]
        )
    if name == "ExtraTreesRegressor":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=20,
                        max_features=1.0,
                        random_state=config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown estimator: {name}")


def _cv_slices(n_rows: int, config: PipelineConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    val_size = config.cv_validation_size
    max_splits = max(1, (n_rows - config.min_train_size) // val_size)
    n_splits = min(config.cv_splits, max_splits)
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for split in range(n_splits, 0, -1):
        val_end = n_rows - (split - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start
        train_start = max(0, train_end - config.rolling_window)
        if train_end - train_start < config.min_train_size:
            continue
        slices.append((np.arange(train_start, train_end), np.arange(val_start, val_end)))
    return slices


def _baseline_predict(name: str, train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    if name == "zero":
        return np.zeros(len(val))
    if name == "last_value":
        return val["balance_lag_1"].to_numpy()
    if name == "seasonal_naive_5":
        return val["balance_lag_5"].to_numpy()
    if name.startswith("rolling_mean_"):
        window = name.split("_")[-1]
        return val[f"balance_rolling_mean_{window}"].to_numpy()
    if name == "day_of_week_mean":
        means = train.groupby("day_of_week")[TARGET_COL].mean()
        fallback = train[TARGET_COL].mean()
        return val["day_of_week"].map(means).fillna(fallback).to_numpy()
    raise ValueError(name)


def evaluate_candidates(
    frame: pd.DataFrame,
    feature_cols: List[str],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    model_data = frame.dropna(subset=[TARGET_COL]).copy()
    needed = ["key_rate"] + feature_cols
    model_data = model_data.dropna(subset=["key_rate"]).reset_index(drop=True)
    cv = _cv_slices(len(model_data), config)
    if not cv:
        raise ValueError("Недостаточно истории для walk-forward проверки.")

    rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    baselines = [
        "zero",
        "last_value",
        "seasonal_naive_5",
        "rolling_mean_5",
        "rolling_mean_10",
        "rolling_mean_20",
        "day_of_week_mean",
    ]
    ml_names = ["Lasso", "ExtraTreesRegressor"]

    for fold, (train_idx, val_idx) in enumerate(cv, start=1):
        train = model_data.iloc[train_idx].copy()
        val = model_data.iloc[val_idx].copy()
        for name in baselines:
            y_pred = _baseline_predict(name, train, val)
            valid = np.isfinite(y_pred)
            metrics = evaluate_predictions(
                val.loc[valid, TARGET_COL], y_pred[valid], val.loc[valid, "key_rate"], config.error_threshold
            )
            rows.append(
                {
                    "fold": fold,
                    "model_name": name,
                    "train_mode": "all_train",
                    "feature_set": "baseline",
                    **metrics,
                }
            )
            pred_rows.append(pd.DataFrame({"Date": val.loc[valid, "Date"], "actual": val.loc[valid, TARGET_COL], "predicted": y_pred[valid], "model_name": name, "fold": fold}))

        for name in ml_names:
            for train_mode in ["all_train", "normal_train"]:
                fit_train = train
                if train_mode == "normal_train" and "target_outlier_alert" in train:
                    filtered = train.loc[train["target_outlier_alert"].eq(0)]
                    if len(filtered) >= config.min_train_size // 2:
                        fit_train = filtered
                estimator = _make_estimator(name, config)
                estimator.fit(fit_train[feature_cols], fit_train[TARGET_COL])
                y_pred = estimator.predict(val[feature_cols])
                metrics = evaluate_predictions(
                    val[TARGET_COL], y_pred, val["key_rate"], config.error_threshold
                )
                rows.append(
                    {
                        "fold": fold,
                        "model_name": name,
                        "train_mode": train_mode,
                        "feature_set": "all_features",
                        **metrics,
                    }
                )
                pred_rows.append(pd.DataFrame({"Date": val["Date"], "actual": val[TARGET_COL], "predicted": y_pred, "model_name": name, "fold": fold}))

    model_report = pd.DataFrame(rows)
    agg = (
        model_report.groupby(["model_name", "train_mode", "feature_set"], as_index=False)
        .agg(
            business_loss_mean=("business_loss_mean", "mean"),
            business_loss_std=("business_loss_mean", "std"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            share_abs_error_le_0_42_mean=("share_abs_error_le_0_42", "mean"),
            bias_mean_error=("bias_mean_error", "mean"),
        )
        .sort_values(["share_abs_error_le_0_42_mean", "business_loss_mean", "MAE_mean"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    agg["business_loss_rank"] = agg["business_loss_mean"].rank(method="min")
    agg["MAE_rank"] = agg["MAE_mean"].rank(method="min")
    agg["RMSE_rank"] = agg["RMSE_mean"].rank(method="min")
    agg["share_abs_error_le_0_42_rank"] = agg["share_abs_error_le_0_42_mean"].rank(method="min", ascending=False)

    best_share = agg["share_abs_error_le_0_42_mean"].max()
    shortlist = agg.loc[agg["share_abs_error_le_0_42_mean"] >= best_share - 0.02].copy()
    shortlist["combined_rank"] = (
        shortlist["business_loss_rank"] + shortlist["MAE_rank"] + shortlist["RMSE_rank"] + shortlist["share_abs_error_le_0_42_rank"]
    )
    selected_idx = shortlist.sort_values(["combined_rank", "business_loss_mean", "MAE_mean"]).index[0]
    agg["selected"] = False
    agg.loc[selected_idx, "selected"] = True
    agg["comment"] = np.where(
        agg["selected"],
        "Выбрана по максимальной/почти максимальной доле попаданий в 0.42 и лучшему балансу рангов.",
        "",
    )
    selected_row = agg.loc[selected_idx].to_dict()
    selected_summary = {
        "selected_model_name": selected_row["model_name"],
        "selected_train_mode": selected_row["train_mode"],
        "selected_feature_set": selected_row["feature_set"],
        "validation_strategy": f"walk-forward, folds={len(cv)}, validation_size={config.cv_validation_size}, rolling_window={config.rolling_window}",
        "business_loss_mean": selected_row["business_loss_mean"],
        "MAE_mean": selected_row["MAE_mean"],
        "RMSE_mean": selected_row["RMSE_mean"],
        "share_abs_error_le_0_42_mean": selected_row["share_abs_error_le_0_42_mean"],
        "selection_reason": (
            "Модель выбрана, потому что она дает лучший баланс business_loss, MAE/RMSE "
            "и сохраняет share_abs_error_le_0_42 близким к максимуму на walk-forward проверке."
        ),
    }
    validation_predictions = pd.concat(pred_rows, ignore_index=True)
    return model_report, agg, selected_summary, validation_predictions


def train_final_model(
    frame: pd.DataFrame, feature_cols: List[str], selected_summary: Dict[str, Any], config: PipelineConfig
) -> Any:
    train_data = frame.dropna(subset=[TARGET_COL]).copy()
    if selected_summary["selected_feature_set"] == "baseline":
        return {"baseline_name": selected_summary["selected_model_name"]}
    if selected_summary["selected_train_mode"] == "normal_train" and "target_outlier_alert" in train_data:
        filtered = train_data.loc[train_data["target_outlier_alert"].eq(0)]
        if len(filtered) >= config.min_train_size // 2:
            train_data = filtered
    estimator = _make_estimator(selected_summary["selected_model_name"], config)
    estimator.fit(train_data[feature_cols], train_data[TARGET_COL])
    return estimator


def _predict_with_model(model: Any, frame: pd.DataFrame, forecast_idx: int, feature_cols: List[str]) -> float:
    row = frame.iloc[[forecast_idx]]
    if isinstance(model, dict):
        train = frame.iloc[:forecast_idx].dropna(subset=[TARGET_COL])
        name = model["baseline_name"]
        return float(_baseline_predict(name, train, row)[0])
    return float(model.predict(row[feature_cols])[0])


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_artifacts(
    config: PipelineConfig,
    model: Any,
    feature_cols: List[str],
    selected_summary: Dict[str, Any],
    model_selection_report: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    train_frame: pd.DataFrame,
) -> str:
    artifacts = Path(config.artifacts_dir)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_" + str(selected_summary["selected_model_name"]).lower()
    run_dir = artifacts / "models" / "runs" / run_id
    latest_dir = artifacts / "models" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")
    (run_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    model_config = {
        **selected_summary,
        "model_run_id": run_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_date_min": str(train_frame["Date"].min().date()),
        "train_date_max": str(train_frame["Date"].max().date()),
    }
    (run_dir / "model_config.json").write_text(json.dumps(_json_safe(model_config), ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing_config = {
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "tax_day_rule": "28th day; if weekend, previous business day",
        "target": TARGET_COL,
        "allow_external_ffill": config.allow_external_ffill,
        "external_max_staleness_days": config.external_max_staleness_days,
    }
    (run_dir / "preprocessing_config.json").write_text(json.dumps(preprocessing_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "outlier_thresholds.json").write_text(json.dumps(_json_safe(thresholds), ensure_ascii=False, indent=2), encoding="utf-8")
    model_selection_report.to_csv(run_dir / "training_report.csv", index=False)
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    card = (
        f"# Model card\n\n"
        f"- run_id: `{run_id}`\n"
        f"- model: `{selected_summary['selected_model_name']}`\n"
        f"- train mode: `{selected_summary['selected_train_mode']}`\n"
        f"- train period: `{model_config['train_date_min']}` - `{model_config['train_date_max']}`\n"
        f"- validation: `{selected_summary['validation_strategy']}`\n"
        f"- reason: {selected_summary['selection_reason']}\n\n"
        "Ограничения: прогноз использует лаги и внешние факторы, доступные на дату прогноза; "
        "экстремальные дни помечаются флагами и не скрываются из отчетности.\n"
    )
    (run_dir / "model_card.md").write_text(card, encoding="utf-8")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return run_id


def load_latest_checkpoint(config: PipelineConfig) -> Optional[Dict[str, Any]]:
    latest = Path(config.artifacts_dir) / "models" / "latest"
    required = ["model.pkl", "feature_cols.json", "model_config.json", "outlier_thresholds.json"]
    if not latest.exists() or not all((latest / name).exists() for name in required):
        return None
    return {
        "dir": latest,
        "model": joblib.load(latest / "model.pkl"),
        "feature_cols": json.loads((latest / "feature_cols.json").read_text(encoding="utf-8")),
        "model_config": json.loads((latest / "model_config.json").read_text(encoding="utf-8")),
        "thresholds": json.loads((latest / "outlier_thresholds.json").read_text(encoding="utf-8")),
    }


def checkpoint_suits_forecast(
    checkpoint: Optional[Dict[str, Any]],
    forecast_date: pd.Timestamp,
    required_train_max: Optional[pd.Timestamp] = None,
) -> bool:
    if checkpoint is None:
        return False
    train_max = pd.to_datetime(checkpoint["model_config"].get("train_date_max"))
    if pd.isna(train_max) or train_max >= forecast_date:
        return False
    if required_train_max is not None and train_max < required_train_max:
        return False
    return True


def forecast_for_date(
    frame: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    model_config: Dict[str, Any],
    config: PipelineConfig,
    forecast_date: pd.Timestamp,
    data_quality_status: str,
    thresholds: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    matches = frame.index[frame["Date"].eq(forecast_date)].tolist()
    if not matches:
        raise ValueError(f"Дата прогноза {forecast_date.date()} не попала в рабочий календарь.")
    idx = matches[0]
    feature_missing = [c for c in feature_cols if c not in frame.columns]
    if feature_missing:
        raise ValueError(f"Не хватает признаков для прогноза: {feature_missing}")
    predicted = _predict_with_model(model, frame, idx, feature_cols)
    row = frame.loc[idx]
    actual = row[TARGET_COL] if pd.notna(row[TARGET_COL]) else np.nan
    abs_error = abs(actual - predicted) if pd.notna(actual) else np.nan
    business_loss = np.nan
    if pd.notna(actual) and pd.notna(row.get("key_rate")):
        business_loss = float(
            calculate_ideal_profit([actual], [row["key_rate"]])[0]
            - calculate_model_profit([actual], [predicted], [row["key_rate"]])[0]
        )
    feature_outlier_count = count_out_of_range_features(row, frame.loc[frame["Date"] < forecast_date], feature_cols)
    target_alert = False
    if pd.notna(actual) and TARGET_COL in thresholds:
        t = thresholds[TARGET_COL]
        target_alert = bool(actual < t["lower"] or actual > t["upper"] or actual < t["extreme_lower"] or actual > t["extreme_upper"])
    comment = "Прогноз с фактом: ошибка и бизнес-метрики рассчитаны." if pd.notna(actual) else "Прогноз без факта: actual_balance и метрики ошибки пока пустые."
    if "warning" in data_quality_status:
        comment += " Есть предупреждения по внешним факторам."
    result = pd.DataFrame(
        [
            {
                "forecast_created_at": datetime.now().isoformat(timespec="seconds"),
                "forecast_for_date": forecast_date.date().isoformat(),
                "predicted_balance": predicted,
                "actual_balance": actual if pd.notna(actual) else np.nan,
                "abs_error": abs_error,
                "abs_error_le_0_42": (abs_error <= config.error_threshold) if pd.notna(abs_error) else np.nan,
                "business_loss": business_loss,
                "model_run_id": model_config.get("model_run_id"),
                "model_type": model_config.get("selected_model_name"),
                "train_date_min": model_config.get("train_date_min"),
                "train_date_max": model_config.get("train_date_max"),
                "is_tax_day_28": int(row.get("is_tax_day_28", 0)),
                "is_tax_window_28": int(row.get("is_tax_window_28", 0)),
                "has_recent_balance_outlier": int(row.get("has_recent_balance_outlier", 0)),
                "feature_outlier_alert": bool(feature_outlier_count >= 3),
                "target_outlier_alert": target_alert if pd.notna(actual) else np.nan,
                "data_quality_status": data_quality_status,
                "comment": comment,
            }
        ]
    )
    return result


def append_log(path: Path, row: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row.copy()
    out.to_csv(path, index=False)


def count_out_of_range_features(row: pd.Series, reference: pd.DataFrame, feature_cols: List[str]) -> int:
    count = 0
    for col in feature_cols:
        values = pd.to_numeric(reference[col], errors="coerce").dropna() if col in reference else pd.Series(dtype=float)
        value = row.get(col)
        if values.empty or pd.isna(value):
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count += int(value < lower or value > upper)
    return count


def build_drift_report(
    forecast_result: pd.DataFrame,
    forecast_log: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    config: PipelineConfig,
) -> pd.DataFrame:
    row = forecast_result.iloc[0].to_dict()
    has_fact = pd.notna(row.get("actual_balance"))
    if not has_fact:
        return pd.DataFrame(
            [
                {
                    "forecast_for_date": row["forecast_for_date"],
                    "has_fact": False,
                    "quality_drift_alert": False,
                    "feature_drift_alert": bool(row.get("feature_outlier_alert", False)),
                    "target_outlier_alert": np.nan,
                    "drift_alert": bool(row.get("feature_outlier_alert", False)),
                    "recommendation": "Ожидать фактический Balance; мониторинг качества будет обновлен после появления факта.",
                }
            ]
        )
    log = forecast_log.copy()
    fact_log = log[pd.notna(log.get("actual_balance"))].copy()
    fact_log["abs_error"] = pd.to_numeric(fact_log["abs_error"], errors="coerce")
    fact_log["business_loss"] = pd.to_numeric(fact_log["business_loss"], errors="coerce")
    rolling_mae = fact_log["abs_error"].tail(config.drift_window).mean()
    rolling_rmse = math.sqrt((fact_log["abs_error"].tail(config.drift_window) ** 2).mean())
    rolling_bl = fact_log["business_loss"].tail(config.drift_window).mean()
    large_error_share = (fact_log["abs_error"].tail(config.drift_window) > config.error_threshold).mean()
    quality_alert = bool(rolling_mae > config.error_threshold or large_error_share > 0.20)
    feature_alert = bool(row.get("feature_outlier_alert", False))
    target_alert = bool(row.get("target_outlier_alert", False))
    drift_alert = quality_alert or feature_alert or target_alert
    recommendation = (
        "Сработал drift_alert: рекомендована внеплановая проверка модели и мониторинг повторных срабатываний."
        if drift_alert
        else "Критичных признаков разладки нет; продолжать плановое ежемесячное дообучение."
    )
    return pd.DataFrame(
        [
            {
                "forecast_for_date": row["forecast_for_date"],
                "has_fact": True,
                "abs_error": row.get("abs_error"),
                "rolling_MAE_20": rolling_mae,
                "rolling_RMSE_20": rolling_rmse,
                "rolling_business_loss_20": rolling_bl,
                "rolling_large_error_share_20": large_error_share,
                "quality_drift_alert": quality_alert,
                "feature_drift_alert": feature_alert,
                "target_outlier_alert": target_alert,
                "drift_alert": drift_alert,
                "recommendation": recommendation,
            }
        ]
    )


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    main_df = read_main_data(config.main_file)
    forecast_date = resolve_forecast_date(main_df, config.forecast_date)
    data_quality_report = validate_main_data(main_df)
    external = read_external_data(config)

    train_main = main_df.loc[main_df["Date"] < forecast_date].copy()
    if len(train_main) < config.min_train_size:
        raise ValueError("Недостаточно истории до FORECAST_DATE для обучения.")
    initial_frame, initial_status = build_model_frame(train_main, external, config, forecast_date)
    initial_train = initial_frame.loc[initial_frame["Date"] < forecast_date].dropna(subset=[TARGET_COL])
    outlier_thresholds = compute_outlier_thresholds(initial_train)
    train_frame, data_quality_status = build_model_frame(train_main, external, config, forecast_date, outlier_thresholds)
    full_frame, full_status = build_model_frame(main_df, external, config, forecast_date, outlier_thresholds)
    if "warning" in full_status and "warning" not in data_quality_status:
        data_quality_status = full_status

    feature_cols = get_feature_cols(train_frame)
    model_report, model_selection_report, selected_model_summary, validation_predictions = evaluate_candidates(
        train_frame, feature_cols, config
    )
    selected_model_summary["train_date_min"] = str(initial_train["Date"].min().date())
    selected_model_summary["train_date_max"] = str(initial_train["Date"].max().date())

    checkpoint = load_latest_checkpoint(config)
    required_train_max = initial_train["Date"].max()
    if checkpoint_suits_forecast(checkpoint, forecast_date, required_train_max):
        final_model = checkpoint["model"]
        final_feature_cols = checkpoint["feature_cols"]
        model_config = checkpoint["model_config"]
        outlier_thresholds = checkpoint["thresholds"]
        full_frame, data_quality_status = build_model_frame(main_df, external, config, forecast_date, outlier_thresholds)
    else:
        final_model = train_final_model(train_frame, feature_cols, selected_model_summary, config)
        run_id = save_artifacts(
            config,
            final_model,
            feature_cols,
            selected_model_summary,
            model_selection_report,
            validation_predictions,
            outlier_thresholds,
            initial_train,
        )
        final_feature_cols = feature_cols
        model_config = {
            **selected_model_summary,
            "model_run_id": run_id,
            "train_date_min": selected_model_summary["train_date_min"],
            "train_date_max": selected_model_summary["train_date_max"],
        }

    forecast_result = forecast_for_date(
        full_frame,
        final_model,
        final_feature_cols,
        model_config,
        config,
        forecast_date,
        data_quality_status,
        outlier_thresholds,
    )
    forecast_log_path = Path(config.artifacts_dir) / "forecasts" / "forecast_log.csv"
    append_log(forecast_log_path, forecast_result)
    forecast_log = pd.read_csv(forecast_log_path)
    drift_report = build_drift_report(forecast_result, forecast_log, outlier_thresholds, config)
    drift_log_path = Path(config.artifacts_dir) / "monitoring" / "drift_log.csv"
    append_log(drift_log_path, drift_report)

    return {
        "config": asdict(config),
        "forecast_date": forecast_date,
        "data_quality_report": data_quality_report,
        "model_report": model_report,
        "model_selection_report": model_selection_report,
        "selected_model_summary": pd.DataFrame([selected_model_summary]),
        "final_model": final_model,
        "final_feature_cols": final_feature_cols,
        "next_day_forecast": forecast_result,
        "forecast_result": forecast_result,
        "drift_report": drift_report,
        "forecast_log": forecast_log.tail(20),
        "train_frame": train_frame,
        "full_frame": full_frame,
    }
