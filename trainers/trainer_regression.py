import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr


class TrainerRegression:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        device=None,
        patience=5,
        save_dir="outputs/models",
        save_name="best_model0116.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.best_val_loss = np.inf
        self.wait = 0

        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, save_name)

    # =====================
    # Train one epoch
    # =====================
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device).float()

            self.optimizer.zero_grad()
            pred = self.model(X).view(-1)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)

        return total_loss / len(self.train_loader.dataset)

    # =====================
    # Validation
    # =====================
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        preds, labels = [], []

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()

                pred = self.model(X).view(-1)
                loss = self.criterion(pred, y)

                total_loss += loss.item() * X.size(0)
                preds.append(pred.cpu().numpy())
                labels.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        val_loss = total_loss / len(self.val_loader.dataset)

        # IC / RankIC
        ic = np.corrcoef(preds, labels)[0, 1]
        if np.isnan(ic):
            ic = 0.0
        rank_ic = spearmanr(preds, labels).correlation
        if rank_ic is None or np.isnan(rank_ic):
            rank_ic = 0.0

        return val_loss, ic, rank_ic, preds, labels

    # =====================
    # Main training loop
    # =====================
    def train(self, epochs):
        best_preds, best_labels = None, None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()

            if self.val_loader is None:
                print(f"Epoch {epoch:03d} | Train Loss {train_loss:.6f}")
                continue

            val_loss, ic, rank_ic, preds, labels = self.validate()

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f} | "
                f"IC {ic:.4f} | RankIC {rank_ic:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                best_preds, best_labels = preds, labels
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("⛔ Early stopping triggered")
                    break

        return best_preds, best_labels

    def load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.to(self.device)
        print(f"✅ Loaded model from {self.best_model_path}")
