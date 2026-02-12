"""
Confidence-based prediction filter.

If the SVM isn't confident enough in its prediction, we fall back
to "background" (class 5 = no action) instead of sending a noisy
command to the robot. This is especially important for Gazebo
where a wrong gesture could drive the robot into a wall.

Low-confidence frames get logged to data/retrain_buffer/ with their
feature vectors, timestamps, and predicted classes. This creates an
active learning loop — you can review these uncertain samples, label
them, and retrain the SVM to improve on edge cases.
"""

import os
import time
import csv
from typing import Tuple, Optional
import numpy as np


class ConfidenceFilter:
    """
    Filters low-confidence predictions by falling back to background class.

    When confidence is below threshold, the prediction gets overridden to
    class 5 (no action) and the frame's features are saved for potential
    retraining. Tracks rejection stats for diagnostics.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        background_class: int = 5,
        log_dir: Optional[str] = None,
    ):
        self.threshold = threshold
        self.background_class = background_class

        # stats
        self.total_predictions = 0
        self.rejected_predictions = 0

        # retrain buffer logging
        if log_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(project_root, 'data', 'retrain_buffer')
        self.log_dir = log_dir
        self._log_file = None
        self._csv_writer = None

    def _ensure_log_file(self):
        """Lazily create the retrain buffer CSV on first rejected frame."""
        if self._csv_writer is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.log_dir, f'uncertain_{timestamp}.csv')
        self._log_file = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._log_file)
        # header: timestamp, predicted_class, confidence, then 13 feature cols
        header = ['timestamp', 'predicted_class', 'confidence']
        header += [f'feat_{i}' for i in range(13)]
        self._csv_writer.writerow(header)

    def filter(self, prediction: int, confidence: float) -> Tuple[int, float]:
        """
        Apply confidence threshold.

        Returns (possibly modified prediction, original confidence).
        If confidence < threshold and it's not background,
        override to background class.
        """
        self.total_predictions += 1

        # always let background through
        if prediction == self.background_class:
            return prediction, confidence

        if confidence < self.threshold:
            self.rejected_predictions += 1
            return self.background_class, confidence

        return prediction, confidence

    def log_uncertain(
        self,
        prediction: int,
        confidence: float,
        features: Optional[np.ndarray] = None,
    ):
        """
        Save a low-confidence sample for potential retraining.
        Called externally when features are available and confidence is low.
        """
        if confidence >= self.threshold:
            return
        if prediction == self.background_class:
            return

        self._ensure_log_file()
        row = [time.time(), prediction, confidence]
        if features is not None:
            row.extend(features.tolist())
        else:
            row.extend([0.0] * 13)
        self._csv_writer.writerow(row)
        self._log_file.flush()

    @property
    def rejection_rate(self) -> float:
        """Fraction of predictions rejected due to low confidence."""
        if self.total_predictions == 0:
            return 0.0
        return self.rejected_predictions / self.total_predictions

    def summary(self) -> dict:
        """Stats dict for the latency/benchmark report."""
        return {
            'total': self.total_predictions,
            'rejected': self.rejected_predictions,
            'rejection_rate': self.rejection_rate,
            'threshold': self.threshold,
        }

    def close(self):
        """Flush and close the retrain log file."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._csv_writer = None
