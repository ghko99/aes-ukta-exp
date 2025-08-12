from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------
MODES = ["baseline", "gru_with_ln", "gru_with_ln_ukta", "gru_with_ln_ukta_attention"]

RUBRIC_NAMES = [
    "exp-grammar",
    "exp-vocab",
    "exp-sentence",
    "org-InterParagraph",
    "org-InParagraph",
    "org-consistency",
    "org-length",
    "cont-clarity",
    "cont-novelty",
    "cont-prompt",        # (skip 대상)
    "cont-description",
]
RUBRIC_SKIP_INDEX = 9     # cont-prompt 는 평균에서 제외

SCORE_MIN, SCORE_MAX = 0, 3
# ---------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------
def round_and_clip(arr: np.ndarray, vmin: int, vmax: int) -> np.ndarray:
    """반올림 후 클리핑하여 정수 스코어로 변환"""
    return np.clip(np.rint(arr), vmin, vmax).astype(int)


def load_pair(mode: str, is_topic_label: bool) -> tuple[np.ndarray, np.ndarray]:
    """y_pred/y_true .npy 로드 (모드+라벨링 조합에 맞춰 경로 생성)"""
    prefix = "topic_" if is_topic_label else "not_topic_"
    base = Path(f"./{mode}")
    y_pred_path = base / f"{prefix}y_pred_.npy"
    y_true_path = base / f"{prefix}y_true_.npy"

    if not y_pred_path.exists() or not y_true_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {y_pred_path} / {y_true_path}")

    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path)
    return y_pred, y_true


# ---------------------------------------------------------------------
# 메트릭
# ---------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, real: np.ndarray) -> float:
    """각 루브릭별 Quadratic Weighted Kappa와 skip 제외 평균을 출력/반환"""
    assert pred.shape == real.shape, "pred/real shape mismatch"

    total = 0.0
    count = 0

    for i, name in enumerate(RUBRIC_NAMES):
        if i == RUBRIC_SKIP_INDEX:
            continue
        kappa = cohen_kappa_score(real[:, i], pred[:, i], weights="quadratic")
        print(f"  - {name:<18}: {kappa: .4f}")
        total += kappa
        count += 1

    avg = total / max(1, count)
    print(f"  → average (skip idx {RUBRIC_SKIP_INDEX}): {avg:.4f}\n")
    return avg


# ---------------------------------------------------------------------
# 평가 루틴
# ---------------------------------------------------------------------
def evaluate(mode: str, is_topic_label: bool) -> float | None:
    label_tag = "prompt" if is_topic_label else "no-prompt"
    print(f"Evaluating mode: {mode} ({label_tag})")

    try:
        pred, real = load_pair(mode, is_topic_label)
    except FileNotFoundError as e:
        print(f"  ! {e}\n")
        return None

    pred = round_and_clip(pred, SCORE_MIN, SCORE_MAX)
    real = round_and_clip(real, SCORE_MIN, SCORE_MAX)

    return compute_metrics(pred, real)


def main() -> None:
    # baseline 은 no-prompt 먼저, 이어서 모든 모드에 대해 prompt(True) 평가
    if "baseline" in MODES:
        evaluate("baseline", is_topic_label=False)

    for mode in MODES:
        evaluate(mode, is_topic_label=True)


if __name__ == "__main__":
    main()
