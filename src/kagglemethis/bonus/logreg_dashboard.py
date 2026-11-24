from functools import wraps
from html import escape
import json
from pathlib import Path
import warnings
from IPython import get_ipython
from IPython.display import HTML, display

import numpy as np


TEMPLATE_PATH = Path(__file__).with_suffix(".html")
OUTPUT_PATH = Path("output/logreg_dashboard.html")
MAX_CHECKPOINTS = 300
GRID_SIZE = 25
PLAYBACK_DURATION_MS = 20_000

def record_training(fit):
    @wraps(fit)
    def wrapped(model, data, target):
        if not getattr(model, "record", True):
            return fit(model, data, target)

        recorder = _TrainingRecorder(data, target, model.epochs)
        model._dashboard_recorder = recorder
        try:
            result = fit(model, data, target)
        finally:
            del model._dashboard_recorder

        output_path = getattr(model, "dashboard_path", None) or OUTPUT_PATH
        try:
            path, document = recorder.save(model, output_path=output_path)
            model.dashboard_path = str(path)
            _display(path, document)
        except (OSError, ValueError) as error:
            print(f"Could not create the LogReg dashboard: {error}")
        return result

    return wrapped


def training_recorder(model):
    return getattr(model, "_dashboard_recorder", None)


def _display(path, document):
    print(f"LogReg dashboard: {path}")
    if get_ipython() is None:
        return

    resize = (
        "const frame=this;const resize=()=>{frame.style.height="
        "Math.ceil(frame.contentDocument.body.getBoundingClientRect().height)+'px';};"
        "new frame.contentWindow.ResizeObserver(resize).observe(frame.contentDocument.body);resize();"
    )
    display(HTML(
        '<iframe title="Visualisasi Logistic Regression" '
        'style="display:block;width:100%;height:960px;border:0" '
        f'onload="{escape(resize, quote=True)}" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    ))


class _TrainingRecorder:
    def __init__(self, data, target, epochs):
        self.feature = 0
        self.feature_name = "feature_0"

        self.data = np.asarray(data, dtype=float)
        self.sign = 1 - 2 * np.asarray(target, dtype=float)
        total_steps = len(data) * epochs

        # Logarithmic spacing captures early updates densely with a fixed budget
        steps = np.geomspace(1, max(1, total_steps), MAX_CHECKPOINTS - 1)
        checkpoints = np.unique(np.r_[0, steps.round().astype(int).clip(0, total_steps)]).tolist()
        self.checkpoints = set(checkpoints)
        self.steps = []
        self.trace = []

    def observe(self, model, step):
        if step not in self.checkpoints:
            return

        scores = self.data @ model.weights + model.bias
        loss = np.logaddexp(0, self.sign * scores).mean()
        # Rows contain fractional epoch, bias, selected feature weight, log-loss

        self.steps.append(step)
        self.trace.append([step / len(self.data), model.bias, model.weights[self.feature], loss])

    def _surface(self, weights, theta0, theta1):
        feature = self.data[:, self.feature]
        offset = self.data @ weights - feature * weights[self.feature]

        bias_grid, weight_grid = np.meshgrid(theta0, theta1)
        bias_grid, weight_grid = bias_grid.ravel(), weight_grid.ravel()

        total = np.zeros(bias_grid.size)
        chunk = max(1, 1_000_000 // bias_grid.size)

        for start in range(0, len(feature), chunk):
            stop = start + chunk
            scores = feature[start:stop, None] * weight_grid
            scores += offset[start:stop, None] + bias_grid
            scores *= self.sign[start:stop, None]

            np.logaddexp(0, scores, out=scores)

            total += scores.sum(axis=0)

        return (total / len(feature)).reshape(GRID_SIZE, GRID_SIZE).round(8).tolist()

    def save(self, model, output_path=OUTPUT_PATH):
        trace = np.asarray(self.trace)

        def axis(values):
            low, high = values.min(), values.max()
            padding = max((high - low) * .5, .5)

            return np.linspace(low - padding, high + padding, GRID_SIZE)

        theta0, theta1 = axis(trace[:, 1]), axis(trace[:, 2])
        payload = {
            "feature": self.feature_name,
            "samples": len(self.data),
            "features": self.data.shape[1],
            "rate": float(model.rate),
            "penalty": model.penalty,
            "strength": float(model.strength),
            "ratio": float(model.ratio),
            "steps": self.steps,
            "playback_duration_ms": PLAYBACK_DURATION_MS,
            "trace": trace.round(8).tolist(),
            "theta0": theta0.round(8).tolist(),
            "theta1": theta1.round(8).tolist(),
            "frames": [{"point": len(trace) - 1, "grid": self._surface(model.weights, theta0, theta1)}]
        }
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
        document = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__TRAINING_DATA__", encoded)

        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")

        return path, document
