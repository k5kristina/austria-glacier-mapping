"""Classical remote sensing baselines: NDSI thresholding and a pixel-wise Random Forest."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class NDSIThresholdModel:
    """NDSI threshold, the classic snow/ice index method; the threshold is tuned on training data."""

    def __init__(self, ndsi_channel_idx: int = 6) -> None:
        self.ndsi_channel_idx = ndsi_channel_idx
        self.best_threshold: float = 0.40
        self.temperature: float = 0.12

    def fit(self, features_val: np.ndarray, masks_val: np.ndarray) -> float:
        """Select the NDSI threshold that maximises pixel IoU on the given data.

        Parameters
        ----------
        features_val : np.ndarray
            Raw (not normalised) feature array of shape ``(N, C, H, W)``.
        masks_val : np.ndarray
            Binary glacier ground-truth masks of shape ``(N, H, W)``.
        """
        ndsi = features_val[:, self.ndsi_channel_idx, :, :]
        target = masks_val.astype(bool)

        best_iou = -1.0
        best_thr = 0.40
        for thr in np.linspace(0.10, 0.65, 56):
            pred = ndsi >= thr
            inter = np.logical_and(pred, target).sum()
            union = np.logical_or(pred, target).sum()
            iou = float(inter / max(union, 1))
            if iou > best_iou:
                best_iou = iou
                best_thr = float(thr)

        self.best_threshold = best_thr
        return self.best_threshold

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return pixel-wise glacier probability map of shape ``(N, H, W)``."""
        ndsi = features[:, self.ndsi_channel_idx, :, :]
        logits = (ndsi - self.best_threshold) / max(self.temperature, 1e-4)
        logits = np.clip(logits, -20.0, 20.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary glacier mask of shape ``(N, H, W)``."""
        return (self.predict_proba(features) >= threshold).astype(np.uint8)


class RandomForestPixelModel:
    """Random Forest on per-pixel spectral, index and terrain features (no spatial context)."""

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 10,
        max_pixels_train: int = 45000,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_pixels_train = max_pixels_train
        self.seed = seed
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    @staticmethod
    def _extract_pixel_table(features: np.ndarray) -> np.ndarray:
        """Flatten ``(N, C, H, W)`` tensor into ``(N*H*W, C)`` pixel feature matrix."""
        n, c, h, w = features.shape
        transposed = np.transpose(features, (0, 2, 3, 1))  # (N, H, W, C)
        return transposed.reshape(n * h * w, c).astype(np.float32)

    def fit(self, features_train: np.ndarray, masks_train: np.ndarray) -> RandomForestPixelModel:
        """Train Random Forest classifier on subsampled pixels from the training set."""
        x_all = self._extract_pixel_table(features_train)
        y_all = masks_train.reshape(-1).astype(np.int64)

        rng = np.random.default_rng(self.seed)
        if x_all.shape[0] > self.max_pixels_train:
            idx = rng.choice(x_all.shape[0], size=self.max_pixels_train, replace=False)
            x_sub = x_all[idx]
            y_sub = y_all[idx]
        else:
            x_sub, y_sub = x_all, y_all

        self.clf.fit(x_sub, y_sub)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict pixel-wise glacier probability map of shape ``(N, H, W)``."""
        n, _, h, w = features.shape
        x_flat = self._extract_pixel_table(features)
        probs = self.clf.predict_proba(x_flat)[:, 1]
        return probs.reshape(n, h, w).astype(np.float32)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary glacier mask of shape ``(N, H, W)``."""
        return (self.predict_proba(features) >= threshold).astype(np.uint8)
