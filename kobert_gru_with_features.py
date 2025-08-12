# ================================================================
#  Embedding‑기반 에세이 채점 모델
#  ─ 수정 내용: heavy I/O(데이터셋·임베딩) 단 1회만 실행
#  ─ 실험 모드: baseline, gru_with_ln, gru_with_ln_ukta, gru_with_ln_ukta_attention
#  ─ 프롬프트 여부 : is_topic_label
# ================================================================
import os, time, json, gc, random
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from psutil import virtual_memory   # 사용 여부는 선택
from embedding import get_essay_dataset_11_rubrics
from config import config, features
from sklearn.preprocessing import StandardScaler
from typing import Literal


# ------------------------- GRUScoreModule (구버전) ------------------------

class GRUScoreModule(nn.Module):
    def __init__(self,output_dim, hidden_dim, ukt_a_dim, dropout=0.5):
        super(GRUScoreModule, self).__init__()
        self.gru = nn.GRU(768,hidden_dim, dropout=dropout, batch_first=True, bidirectional=True)        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, ukta):
        x, _ = self.gru(x)
        
        x = x[:, -1, :]  # Use the output of the last time step
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x , None

# ============================== 단계 1: 개선된 GRU 모델 (layer_norm 추가) ==============================
class GRUScoreModuleWithLN(nn.Module):
    def __init__(self, output_dim, hidden_dim, ukt_a_dim, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ukta):
        x, _ = self.gru(x)  # [B, T, 2H]
        
        # 변경: 마지막 타임스텝 대신 평균 pooling 사용
        x = torch.mean(x, dim=1)  # [B, 2H]
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x) , None

# ========================= 단계 3: 개선된 GRU 모델 + UKTA + No Attention =========================
class GRUScoreModuleWithLNUKTA(nn.Module):
    def __init__(self, output_dim, hidden_dim, ukt_a_dim=294, dropout=0.5):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        
        self.ukt_a_fc = nn.Linear(ukt_a_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ukt_a):
        x, _ = self.gru(x)  # [B, T, 2H]
        
        # 변경: 마지막 타임스텝 대신 평균 pooling 사용
        x = torch.mean(x, dim=1)  # [B, 2H]
        x = self.layer_norm(x)

        ukt_a_features = self.ukt_a_fc(ukt_a)
        combined = torch.cat((x, ukt_a_features), dim=1)  # [B, 3H]

        combined = self.dropout(combined)
        combined = self.fc(combined)
        output = self.sigmoid(combined)
        
        return output, None

# ========================= 단계 4: 개선된 GRU 모델 + UKTA + Attention =========================
class GRUScoreModuleWithLNUKTAAttention(nn.Module):
    def __init__(self, output_dim, hidden_dim, ukt_a_dim=294, dropout=0.5):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        
        self.ukt_a_fc = nn.Linear(ukt_a_dim, hidden_dim)
        
        # UKT-A attention을 위한 단순한 레이어
        self.attention_weights = nn.Linear(hidden_dim * 2, ukt_a_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ukt_a):
        # GRU 처리
        x, _ = self.gru(x)  # [B, T, 2H]
        x = torch.mean(x, dim=1)  # [B, 2H]
        x = self.layer_norm(x)

        # UKT-A attention 계산 (GRU output 기반)
        attention_scores = self.attention_weights(x)  # [B, ukt_a_dim]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, ukt_a_dim]
        
        # Attention을 적용한 UKT-A features
        weighted_ukt_a = ukt_a * attention_weights  # [B, ukt_a_dim]
        ukt_a_features = self.ukt_a_fc(weighted_ukt_a)  # [B, H]
        
        # Combined features
        combined = torch.cat((x, ukt_a_features), dim=1)  # [B, 3H]
        combined = self.dropout(combined)
        combined = self.fc(combined)
        output = self.sigmoid(combined)
        
        return output, attention_weights

# ------------------------- Dataset 클래스 ------------------------
class EssayDataset(Dataset):
    def __init__(self, embedded_essays, ukta_features, labels, maxlen=128):
        self.embedded_essays = embedded_essays
        self.ukta_features = torch.tensor(ukta_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        essay = self.embedded_essays[idx]
        if hasattr(essay, "values"):          # DataFrame → np.array
            essay = essay.values
        if len(essay) < self.maxlen:          # pre‑padding
            padded = np.zeros((self.maxlen, essay.shape[1]), dtype=np.float32)
            padded[-len(essay) :] = essay
        else:                                 # truncate
            padded = essay[: self.maxlen].astype(np.float32)
        return torch.tensor(padded), self.ukta_features[idx], self.labels[idx]

# ------------------------ 유틸리티 함수 -------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ------------------- 임베딩 분할 (I/O 제거 버전) -------------------
def get_embedded_essay(train_index, valid_index, essays, embedded_df):
    essay_ranges, start = [], 0
    for essay in essays:
        end = start + len(essay)
        essay_ranges.append((start, end))
        start = end
    valid_set = set(valid_index)
    train_set = set(train_index)
    train_embs = [
        embedded_df.iloc[s:e] for ix, (s, e) in enumerate(essay_ranges) if ix in train_set
    ]
    valid_embs = [
        embedded_df.iloc[s:e] for ix, (s, e) in enumerate(essay_ranges) if ix in valid_set
    ]
    return train_embs, valid_embs

# ---------------------------- Main ------------------------------
def main(args, dataset, essays, y, embedded_df):
    # 하이퍼파라미터
    dropout = 0.30475784381583626
    lr = 9.154637004201508e-4
    n_epochs = 100
    hidden_dim = 128
    batch_size = 128
    seed = 42
    patience = 10
    n_outputs = y.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    mode = config["mode"]    

    if not os.path.exists(mode):
        os.makedirs(mode)

    #features 없는 rows 제외
    nonzero_mask = (dataset[features].sum(axis=1) != 0)
    # NaN을 0으로 보고 합치고 싶다면:
    # nonzero_mask = (dataset[features].fillna(0).sum(axis=1) != 0)

    train_idx = dataset.index[dataset['is_train']]
    valid_idx = dataset.index[~dataset['is_train']]

    # 각 split에 마스크 적용 (순서 유지)
    train_idx = train_idx[nonzero_mask.loc[train_idx].values]
    valid_idx = valid_idx[nonzero_mask.loc[valid_idx].values]

    # 필터링된 feature 행렬
    train_features = dataset.loc[train_idx, features].to_numpy()
    valid_features = dataset.loc[valid_idx, features].to_numpy()

    # Feature scaling
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)

    print(f"Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
    print(f"Train features shape: {np.array(train_features).shape}, Valid features shape: {np.array(valid_features).shape}")
    # ---------- 임베딩 / 라벨 분할 ----------
    train_embs, valid_embs = get_embedded_essay(train_idx, valid_idx, essays, embedded_df)
    y_train, y_valid = y[train_idx], y[valid_idx]
    print(f"Train emb size: {len(train_embs)}, Valid emb size: {len(valid_embs)}")
    # ---------- DataLoader ----------
    train_loader = DataLoader(
        EssayDataset(train_embs, train_features, y_train), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        EssayDataset(valid_embs, valid_features, y_valid), batch_size=batch_size, shuffle=False
    )

    # ---------- 모델 학습 ----------

    if config["mode"] == "baseline":
        model = GRUScoreModule(n_outputs, hidden_dim, ukt_a_dim=294, dropout=dropout).to(device)
    elif config["mode"] == "gru_with_ln":
        model = GRUScoreModuleWithLN(n_outputs, hidden_dim, ukt_a_dim=294, dropout=dropout).to(device)
    elif config["mode"] == "gru_with_ln_ukta":
        model = GRUScoreModuleWithLNUKTA(n_outputs, hidden_dim, ukt_a_dim=294, dropout=dropout).to(device)
    elif config["mode"] == "gru_with_ln_ukta_attention":
        model = GRUScoreModuleWithLNUKTAAttention(n_outputs, hidden_dim, ukt_a_dim=294, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_outputs = None
    best_attention = None 
    early_stop = 0
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, ub, yb in train_loader:
            xb, ub, yb = xb.to(device), ub.to(device), yb.to(device)
            optimizer.zero_grad()
            output, _ = model(xb, ub)
            loss = criterion(output, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, outputs = 0.0, []
        attention = []
        with torch.no_grad():
            for xb, ub, yb in valid_loader:
                xb, ub, yb = xb.to(device), ub.to(device), yb.to(device)
                out, attn = model(xb, ub)
                val_loss += criterion(out, yb).item()
                outputs.append(out.cpu().numpy())
                attention.append(attn.cpu().numpy() if attn is not None else None)

        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)

        print(
            f"[{epoch:03d}/{n_epochs}] "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Δt {time.time() - start_time:.1f}s"
        )
        start_time = time.time()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_outputs = np.vstack(outputs)
            torch.save(
                model.state_dict(),
                f"./{mode}/{ 'topic_' if config['is_topic_label'] else 'not_topic_' }model.pth"
            )

            if attention[0] is not None:
                best_attention = np.vstack(attention)

            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                print("Early stopping.")
                break

    # ---------- 평가 ----------
    y_pred = best_outputs * args["num_range"]
    y_true = np.array(y_valid) * args["num_range"]


    np.save(f'./{mode}/{"topic_" if config["is_topic_label"] else "not_topic_"}y_true_.npy', y_true)
    np.save(f'./{mode}/{"topic_" if config["is_topic_label"] else "not_topic_"}y_pred_.npy', y_pred)

    if best_attention is not None:
        np.save(f"./{mode}/attention.npy", best_attention)


    torch.cuda.empty_cache()
    gc.collect()
    del model, optimizer, criterion, train_loader, valid_loader
    print("Memory usage after training:", virtual_memory().percent)
    return y_pred, y_true

# ----------------------- 진입점 -----------------------



if __name__ == "__main__":
    # 0) 공통 설정
    args = config["aihub_v1"]        # 필요 시 다른 데이터셋으로 교체
    mode = "gru_with_ln_ukta_attention" # baseline, gru_with_ln, gru_with_ln_ukta, gru_with_ln_ukta_attention

    # Experiment 세팅
    config["mode"] = mode
    config["is_topic_label"] = True # True, False

    # 1) heavy I/O (딱 한 번)
    DATASET = pd.read_csv(args["dataset_path"], encoding="utf-8-sig")
    ESSAYS, Y = get_essay_dataset_11_rubrics(is_rubric=True, args=args)

    emb_file = os.path.join(
        args["emb_file_path"],
        f"{args['dataset_path'].split('/')[1]}_{'notlabeled' if not config['is_topic_label'] else 'labeled'}.csv",
    )
    print(emb_file)
    EMBEDDED_DF = dd.read_csv(emb_file, encoding="cp949", header=None).compute()
    
    main(
        args, DATASET, ESSAYS, Y, EMBEDDED_DF
    )
