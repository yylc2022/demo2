import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=0.001,
        device=None,
        patience=5,
        save_dir="outputs/models",
        save_name="best_model0116.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.patience = patience
        self.save_dir = save_dir
        self.save_name = save_name
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        # EarlyStopping
        self.best_ic = -np.inf
        self.wait = 0
        self.best_model_path = os.path.join(self.save_dir, self.save_name)

    # =====================
    # Train one epoch
    # =====================
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(X).squeeze()
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)

        return total_loss / len(self.train_loader.dataset)

    # =====================
    # Validation
    # =====================
    def validate(self):
        self.model.eval()

        total_loss = 0
        probs = []
        labels = []

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()

                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)

                #prob = torch.sigmoid(logits)
                prob=logits

                total_loss += loss.item() * X.size(0)
                probs.append(prob.cpu().numpy())
                labels.append(y.cpu().numpy())

        probs = np.concatenate(probs)
        labels = np.concatenate(labels)

        val_loss = total_loss / len(self.val_loader.dataset)

        # IC / RankIC
        ic = np.corrcoef(probs, labels)[0, 1]
        rank_ic = spearmanr(probs, labels).correlation

        return val_loss, ic, rank_ic, probs, labels

    # =====================
    # Main train loop
    # =====================
    def train(self, epochs):
        best_probs, best_labels = None, None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()

            if self.val_loader is not None:
                val_loss, ic, rank_ic, probs, labels = self.validate()

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss {train_loss:.8f} | "
                    f"Val Loss {val_loss:.8f} | "
                    f"IC {ic:.4f} | RankIC {rank_ic:.4f}"
                )

                # ---- 基于最高的 IC 保存模型（保存文件名带上指标以便留存“一批”候选） ----
                if ic > self.best_ic:
                    self.best_ic = ic
                    self.wait = 0
                    # 包含 epoch 和 ic 到文件名中
                    file_ext = os.path.splitext(self.save_name)[1]
                    file_base = os.path.splitext(self.save_name)[0]

                    # 1. 保存带版本号的模型 (Backup)
                    current_save_name = f"{file_base}_epoch{epoch:03d}_ic_{ic:.4f}{file_ext}"
                    versioned_path = os.path.join(self.save_dir, current_save_name)
                    torch.save(self.model.state_dict(), versioned_path)

                    # 2. 保存/覆盖主模型文件 (Main)
                    self.best_model_path = os.path.join(self.save_dir, self.save_name)
                    torch.save(self.model.state_dict(), self.best_model_path)

                    best_probs, best_labels = probs, labels
                    print(f"✅ Saved best model to {self.best_model_path} (and backup {versioned_path})")
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print("⛔ Early stopping triggered")
                        break
            else:
                print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f}")

        return best_probs, best_labels

    # =====================
    # Load best model
    # =====================
    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            self.model.to(self.device)
            print(f"✅ Loaded best model from {self.best_model_path}")
        else:
            print("⚠️ No saved model found.")
