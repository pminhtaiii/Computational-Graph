from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_FILE = ROOT_DIR / "models" / "model.pth"


def load_model_state():
    if not CHECKPOINT_FILE.exists():
        raise SystemExit("Không tìm thấy mô hình. Hãy chạy train.py trước.")

    state = torch.load(CHECKPOINT_FILE, map_location="cpu")
    coef = state.get("coef", state.get("W"))
    offset = state.get("offset", state.get("b"))

    if coef is None or offset is None:
        raise SystemExit("File mô hình không hợp lệ.")

    return coef, offset


def forecast_final_score():
    coef, offset = load_model_state()

    while True:
        raw_value = input("Điểm giữa kỳ (q để thoát): ").strip()
        if raw_value.lower() in {"q", "exit"}:
            break

        try:
            mid_score = float(raw_value)
        except ValueError:
            continue

        if not 0.0 <= mid_score <= 10.0:
            continue

        sample = torch.tensor([[mid_score]], dtype=torch.float32)
        predicted = (sample @ coef + offset).item()
        predicted = max(0.0, min(10.0, predicted))
        print(f"Điểm cuối kỳ dự đoán: {predicted:.2f}")


if __name__ == "__main__":
    forecast_final_score()
