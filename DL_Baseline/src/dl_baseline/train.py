import copy
import torch

from .metrics import nasa_score


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    criterion = torch.nn.MSELoss()

    best_model = None
    best_val_loss = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_nasa = nasa_score(targets, preds)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val NASA: {val_nasa:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model
