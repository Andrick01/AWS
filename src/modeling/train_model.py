"""LightGBM / XGBoost Pairwise Classifier for Business Entity Resolution.

Trains a gradient-boosted tree classifier on pairwise similarity features
to predict match probability. Optimized for F_0.5 which rewards precision
more than recall per competition guidelines.
"""

import argparse
import csv
import gc
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout support for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature columns used for training (must match build_features.py output exactly).
# Country is handled generically via country_match / country_mismatch — no
# country-specific columns exist, so the model handles any unseen country without
# retraining or code changes.
FEATURE_COLS = [
    "name_exact_match", "name_seq_ratio", "name_jaro_winkler",
    "name_levenshtein_sim", "name_token_jaccard", "name_token_overlap_ratio",
    "name_prefix_match_3", "name_prefix_match_5", "name_len_diff_ratio",
    "addr_exact_match", "addr_seq_ratio", "addr_token_jaccard",
    "addr_token_overlap_ratio", "addr_numeric_exact_match", "addr_numeric_overlap_ratio",
    "addr_has_both_address",
    "country_match", "country_mismatch",
    "is_source2", "is_source3",
]


def load_feature_matrix(tsv_path: Path, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load feature TSV into X (features), y (labels), and metadata columns."""
    logger.info("Loading feature matrix from %s...", tsv_path)
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={col: np.float32 for col in FEATURE_COLS},
        nrows=max_rows,
    )
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    X = df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y = df["label"].values.astype(np.int32) if "label" in df.columns else np.zeros(len(df), dtype=np.int32)

    return df, X, y


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
) -> object:
    """Train a LightGBM binary classifier."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed. Install with: pip install lightgbm")
        raise

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        # Handle class imbalance
        "is_unbalance": True,
    }
    if params:
        default_params.update(params)

    logger.info("Training LightGBM with params: %s", {k: v for k, v in default_params.items() if k != "verbose"})
    logger.info("Train shape: X=%s, pos=%d, neg=%d", X_train.shape, int(y_train.sum()), int(len(y_train) - y_train.sum()))

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    callbacks = [lgb.log_evaluation(50)]

    valid_sets = [train_data]
    valid_names = ["train"]
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("val")
        logger.info("Val shape: X=%s, pos=%d, neg=%d", X_val.shape, int(y_val.sum()), int(len(y_val) - y_val.sum()))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    logger.info("LightGBM training complete. Best iteration: %d", model.best_iteration)

    # Log feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    logger.info("Top 10 features by gain:")
    for fname, imp in feat_imp[:10]:
        logger.info("  %s: %.1f", fname, imp)

    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
) -> object:
    """Train an XGBoost binary classifier as fallback."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Install with: pip install xgboost")
        raise

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "scale_pos_weight": float(pos_weight),
        "seed": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    logger.info("Training XGBoost with scale_pos_weight=%.2f", pos_weight)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    evals = [(dtrain, "train")]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
        evals.append((dval, "val"))

    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        verbose_eval=50,
    )

    logger.info("XGBoost training complete.")
    return model


class EnsembleModel:
    """Ensemble model combining LightGBM (MIT License) and XGBoost (Apache 2.0 License)."""

    def __init__(self, lgb_model: object, xgb_model: object, lgb_weight: float = 0.5):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.lgb_weight = lgb_weight
        self.xgb_weight = 1.0 - lgb_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        s_lgb = predict_scores(self.lgb_model, X)
        s_xgb = predict_scores(self.xgb_model, X)
        return self.lgb_weight * s_lgb + self.xgb_weight * s_xgb


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    lgb_weight: float = 0.5,
) -> EnsembleModel:
    """Train both LightGBM and XGBoost models and return a 2-model ensemble."""
    logger.info("==================================================")
    logger.info("  TRAINING 2-MODEL ENSEMBLE (LightGBM + XGBoost)  ")
    logger.info("==================================================")
    logger.info("Model 1: LightGBM (License: MIT)")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

    logger.info("Model 2: XGBoost (License: Apache 2.0)")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    logger.info("Ensembling LightGBM (weight=%.2f) + XGBoost (weight=%.2f)", lgb_weight, 1.0 - lgb_weight)
    return EnsembleModel(lgb_model, xgb_model, lgb_weight=lgb_weight)


def save_model(model: object, output_path: Path) -> None:
    """Serialize trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", output_path)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle EnsembleModel references cleanly across modules."""
    def find_class(self, module, name):
        if name == "EnsembleModel":
            return EnsembleModel
        return super().find_class(module, name)


def load_model(model_path: Path) -> object:
    """Deserialize trained model from disk."""
    with open(model_path, "rb") as f:
        model = CustomUnpickler(f).load()
    logger.info("Model loaded from %s", model_path)
    return model


def predict_scores(model: object, X: np.ndarray) -> np.ndarray:
    """Generate match probability scores using trained model or ensemble."""
    if hasattr(model, "lgb_model") and hasattr(model, "xgb_model"):
        return model.predict(X)

    try:
        import lightgbm as lgb
        if isinstance(model, lgb.Booster):
            return model.predict(X)
    except ImportError:
        pass

    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X, feature_names=FEATURE_COLS)
            return model.predict(dmat)
    except ImportError:
        pass

    raise ValueError("Unknown model type. Expected LightGBM Booster, XGBoost Booster, or EnsembleModel.")


def main():
    parser = argparse.ArgumentParser(description="Train Entity Resolution Classifier / Ensemble")
    parser.add_argument("--train-features", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_features.tsv")
    parser.add_argument("--val-features", type=Path, default=None, help="Optional validation feature set")
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "models" / "final_model" / "model.pkl")
    parser.add_argument("--engine", type=str, choices=["ensemble", "lightgbm", "xgboost"], default="ensemble")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    args = parser.parse_args()

    # Load training data
    train_df, X_train, y_train = load_feature_matrix(args.train_features, max_rows=args.max_train_rows)
    del train_df
    gc.collect()

    # Load validation data if provided
    X_val, y_val = None, None
    if args.val_features and args.val_features.exists():
        val_df, X_val, y_val = load_feature_matrix(args.val_features, max_rows=args.max_val_rows)
        del val_df
        gc.collect()

    # Train
    if args.engine == "ensemble":
        model = train_ensemble(X_train, y_train, X_val, y_val)
    elif args.engine == "lightgbm":
        model = train_lightgbm(X_train, y_train, X_val, y_val)
    else:
        model = train_xgboost(X_train, y_train, X_val, y_val)

    # Save
    save_model(model, args.model_output)


if __name__ == "__main__":
    main()
