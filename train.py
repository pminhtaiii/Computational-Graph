from pathlib import Path

import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = ROOT_DIR / "dataset" / "TRAIN2.xlsx"
EXPORT_FILE = ROOT_DIR / "models" / "model.pth"


def fit_score_model():
    EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCE_FILE.exists():
        raise SystemExit(f"Không tìm thấy dữ liệu tại: {SOURCE_FILE}")

    frame = pd.read_excel(SOURCE_FILE)
    if not {"midterm", "final"}.issubset(frame.columns):
        raise SystemExit("Dữ liệu cần có hai cột midterm và final.")

    features = torch.tensor(frame["midterm"].to_numpy(), dtype=torch.float32).view(-1, 1)
    targets = torch.tensor(frame["final"].to_numpy(), dtype=torch.float32).view(-1, 1)

    coef = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    offset = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)

    rounds = 1000
    step = 0.01

    for _ in range(rounds):
        forecast = features @ coef + offset
        objective = ((forecast - targets) ** 2).mean()
        objective.backward()

        with torch.no_grad():
            coef -= step * coef.grad
            offset -= step * offset.grad
            coef.grad.zero_()
            offset.grad.zero_()

    torch.save({"coef": coef.detach(), "offset": offset.detach()}, EXPORT_FILE)


if __name__ == "__main__":
    fit_score_model()
