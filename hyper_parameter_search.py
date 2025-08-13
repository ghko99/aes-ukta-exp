# ================================================================
#  Optuna를 사용한 하이퍼파라미터 탐색 - GRUScoreModuleWithLNUKTAAttention
#  ─ 베이지안 최적화로 효율적인 탐색
#  ─ Cohen's Kappa 기반 평가 (높을수록 좋음)
#  ─ 매 Trial마다 Best Parameter 자동 저장
#  ─ 실시간 진행 상황 모니터링
# ================================================================

import os, time, json, gc, random
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from psutil import virtual_memory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from embedding import get_essay_dataset_11_rubrics
from config import config, features

# Rubric names for evaluation
rubric_names = [
    "exp-grammar",
    "exp-vocab", 
    "exp-sentence",
    "org-InterParagraph",
    "org-InParagraph",
    "org-consistency",
    "org-length",
    "cont-clarity",
    "cont-novelty",
    "cont-prompt",
    "cont-description"
]

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
        self.attention_weights = nn.Linear(hidden_dim * 2, ukt_a_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ukt_a):
        x, _ = self.gru(x)
        x = torch.mean(x, dim=1)
        x = self.layer_norm(x)

        attention_scores = self.attention_weights(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        weighted_ukt_a = ukt_a * attention_weights
        ukt_a_features = self.ukt_a_fc(weighted_ukt_a)
        
        combined = torch.cat((x, ukt_a_features), dim=1)
        combined = self.dropout(combined)
        combined = self.fc(combined)
        output = self.sigmoid(combined)
        
        return output, attention_weights

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
        if hasattr(essay, "values"):
            essay = essay.values
        if len(essay) < self.maxlen:
            padded = np.zeros((self.maxlen, essay.shape[1]), dtype=np.float32)
            padded[-len(essay) :] = essay
        else:
            padded = essay[: self.maxlen].astype(np.float32)
        return torch.tensor(padded), self.ukta_features[idx], self.labels[idx]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def compute_metrics(pred, real):
    """
    Cohen's Kappa 기반 평가 메트릭 (높을수록 좋음)
    9번째 인덱스(cont-prompt)는 제외하고 계산
    """
    # 정수로 변환 및 클리핑
    pred = np.rint(pred).astype(int)
    real = np.rint(real).astype(int)
    pred = np.clip(pred, 0, 3)
    real = np.clip(real, 0, 3)
    
    kappas = 0
    kappa_scores = []
    
    for i in range(pred.shape[1]):
        if i == 9:  # cont-prompt 제외
            continue
        kappa = cohen_kappa_score(real[:, i], pred[:, i], weights='quadratic')
        kappa_scores.append(kappa)
        kappas += kappa
    
    average_kappa = kappas / (pred.shape[1] - 1)
    
    return {
        'average_kappa': average_kappa,
        'individual_kappas': kappa_scores,
        'pred': pred,
        'real': real
    }

def get_prepared_data(args, dataset, essays, y, embedded_df):
    """데이터 준비 함수 - 한 번만 실행"""
    nonzero_mask = (dataset[features].sum(axis=1) != 0)
    train_idx = dataset.index[dataset['is_train']]
    valid_idx = dataset.index[~dataset['is_train']]
    train_idx = train_idx[nonzero_mask.loc[train_idx].values]
    valid_idx = valid_idx[nonzero_mask.loc[valid_idx].values]
    
    train_features = dataset.loc[train_idx, features].to_numpy()
    valid_features = dataset.loc[valid_idx, features].to_numpy()
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)
    
    train_embs, valid_embs = get_embedded_essay(train_idx, valid_idx, essays, embedded_df)
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    return {
        'train_embs': train_embs,
        'valid_embs': valid_embs, 
        'train_features': train_features,
        'valid_features': valid_features,
        'y_train': y_train,
        'y_valid': y_valid,
        'scaler': scaler
    }

def objective(trial, data, args):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    patience = trial.suggest_int('patience', 5, 20)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_outputs = data['y_train'].shape[1]
    n_epochs = 100  # Fixed
    
    # Create DataLoaders
    train_loader = DataLoader(
        EssayDataset(data['train_embs'], data['train_features'], data['y_train']), 
        batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        EssayDataset(data['valid_embs'], data['valid_features'], data['y_valid']), 
        batch_size=batch_size, shuffle=False
    )
    
    # Model
    model = GRUScoreModuleWithLNUKTAAttention(
        n_outputs, hidden_dim, ukt_a_dim=294, dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_kappa = -1.0  # Kappa는 높을수록 좋음
    best_outputs = None
    early_stop = 0
    
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
        with torch.no_grad():
            for xb, ub, yb in valid_loader:
                xb, ub, yb = xb.to(device), ub.to(device), yb.to(device)
                out, _ = model(xb, ub)
                val_loss += criterion(out, yb).item()
                outputs.append(out.cpu().numpy())
        
        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)
        
        # Calculate Kappa score
        y_pred = np.vstack(outputs) * args["num_range"]
        y_true = np.array(data['y_valid']) * args["num_range"]
        metrics = compute_metrics(y_pred, y_true)
        current_kappa = metrics['average_kappa']
        
        # Report intermediate value for pruning
        trial.report(current_kappa, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if current_kappa > best_kappa:
            best_kappa = current_kappa
            best_outputs = np.vstack(outputs)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                break
    
    # Clean up
    del model, optimizer, criterion, train_loader, valid_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_kappa

class OptunaHyperparameterSearcher:
    def __init__(self, n_trials=100, study_name="gru_optimization", save_dir="optuna_results"):
        self.n_trials = n_trials
        self.study_name = study_name
        self.save_dir = save_dir
        self.study = None
        self.best_value_history = []  # Track best value over trials
        
    def search(self, args, dataset, essays, y, embedded_df):
        """Run Optuna hyperparameter search"""
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{self.save_dir}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create subdirectories for organized storage
        os.makedirs(f"{self.save_dir}/trial_history", exist_ok=True)
        os.makedirs(f"{self.save_dir}/best_params_history", exist_ok=True)
        
        print("Preparing data...")
        # Prepare data once
        data = get_prepared_data(args, dataset, essays, y, embedded_df)
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            direction='maximize',  # Kappa는 높을수록 좋음
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )
        
        print(f"Starting Optuna optimization with {self.n_trials} trials...")
        print("Objective: Maximize Average Cohen's Kappa")
        print(f"Results will be saved to: {self.save_dir}")
        print("-" * 60)
        
        # Optimize
        self.study.optimize(
            lambda trial: objective(trial, data, args),
            n_trials=self.n_trials,
            callbacks=[self._callback]
        )
        
        # Save and analyze final results
        self.save_final_results()
        self.analyze_results()
        self.create_visualizations()
        
        print(f"\nOptimization completed! Results saved to {self.save_dir}")
        
    def _callback(self, study, trial):
        """Enhanced callback function - saves best params after each trial"""
        
        # Save trial information
        trial_info = {
            'trial_number': trial.number,
            'state': str(trial.state),
            'value': trial.value if trial.value is not None else None,
            'params': trial.params if hasattr(trial, 'params') else None,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
        }
        
        # Save individual trial info
        with open(f"{self.save_dir}/trial_history/trial_{trial.number:04d}.json", 'w') as f:
            json.dump(trial_info, f, indent=2)
        
        # If this trial completed successfully
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Check if this is a new best
            if study.best_value and (not self.best_value_history or study.best_value > max(self.best_value_history)):
                self.best_value_history.append(study.best_value)
                
                # Save best parameters with timestamp
                best_params_info = {
                    'trial_number': study.best_trial.number,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials_so_far': len(study.trials),
                    'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'improvement_from_previous': study.best_value - self.best_value_history[-2] if len(self.best_value_history) > 1 else None
                }
                
                # Save to timestamped file
                filename = f"{self.save_dir}/best_params_history/best_at_trial_{study.best_trial.number:04d}.json"
                with open(filename, 'w') as f:
                    json.dump(best_params_info, f, indent=2)
                
                # Also update the main best_params.json file
                with open(f"{self.save_dir}/best_params_current.json", 'w') as f:
                    json.dump(best_params_info, f, indent=2)
                
                print(f"  🎯 New best found at Trial {study.best_trial.number}: Kappa = {study.best_value:.6f}")
                print(f"     Parameters: {study.best_params}")
        
        # Save study progress every 5 trials
        if trial.number % 5 == 0:
            self.save_study_progress()
            
        # Print progress every trial
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number:3d}: Kappa = {trial.value:.6f} | Best: {study.best_value:.6f} | Complete: {n_complete}, Pruned: {n_pruned}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"Trial {trial.number:3d}: PRUNED | Best: {study.best_value:.6f} | Complete: {n_complete}, Pruned: {n_pruned}")
    
    def save_study_progress(self):
        """Save current study progress"""
        if self.study is None:
            return
            
        # Save trials dataframe
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(f"{self.save_dir}/trials_progress.csv", index=False)
        
        # Save study statistics
        stats = {
            'n_trials_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_trials_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_trials_failed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'current_best_value': self.study.best_value if self.study.best_value else None,
            'current_best_params': self.study.best_params if self.study.best_params else None,
            'current_best_trial': self.study.best_trial.number if self.study.best_trial else None,
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{self.save_dir}/study_progress.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def save_final_results(self):
        """Save final optimization results"""
        if self.study is None:
            return
            
        # Save final study object using pickle for potential reload
        import pickle
        with open(f"{self.save_dir}/study.pkl", 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save final trials DataFrame
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(f"{self.save_dir}/trials_final.csv", index=False)
        
        # Save final best parameters
        best_params = {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'optimization_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{self.save_dir}/best_params_final.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save complete history of best values
        history_df = pd.DataFrame({
            'trial': range(len(self.best_value_history)),
            'best_value': self.best_value_history
        })
        history_df.to_csv(f"{self.save_dir}/best_value_history.csv", index=False)
        
        # Save study summary
        study_summary = {
            'study_name': self.study.study_name,
            'direction': 'maximize',
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number,
            'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{self.save_dir}/study_summary.json", 'w') as f:
            json.dump(study_summary, f, indent=2)
    
    def analyze_results(self):
        """Analyze optimization results"""
        if self.study is None:
            return
            
        print(f"\n{'='*60}")
        print(f"{'OPTIMIZATION RESULTS':^60}")
        print(f"{'='*60}")
        print(f"Best Average Kappa: {self.study.best_value:.6f}")
        print(f"Best Trial: #{self.study.best_trial.number}")
        print(f"Total Trials: {len(self.study.trials)}")
        print(f"  - Completed: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  - Pruned: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  - Failed: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        print(f"\n{'BEST PARAMETERS':^60}")
        print("-" * 60)
        for key, value in self.study.best_params.items():
            print(f"  {key:20s}: {value}")
        
        # Feature importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            print(f"\n{'PARAMETER IMPORTANCE':^60}")
            print("-" * 60)
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_importance:
                bar_length = int(value * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  {key:20s}: {bar} {value:.4f}")
        except:
            print("\nCould not calculate parameter importance (need more trials)")
        
        print("=" * 60)
    
    def create_visualizations(self):
        """Create Optuna visualization plots"""
        if self.study is None:
            return
            
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create optimization history plot manually
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                trial_numbers = [t.number for t in completed_trials]
                values = [t.value for t in completed_trials]
                best_values = [max(values[:i+1]) for i in range(len(values))]
                
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=('Optimization History', 'Best Value Progress'),
                                   vertical_spacing=0.15)
                
                # All trials
                fig.add_trace(
                    go.Scatter(x=trial_numbers, y=values, 
                              mode='markers', name='Trial Values',
                              marker=dict(size=8, color='lightblue')),
                    row=1, col=1
                )
                
                # Best value line
                fig.add_trace(
                    go.Scatter(x=trial_numbers, y=best_values,
                              mode='lines', name='Best Value',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
                
                # Best value progress
                fig.add_trace(
                    go.Scatter(x=trial_numbers, y=best_values,
                              mode='lines+markers', name='Best Kappa',
                              line=dict(color='green', width=2)),
                    row=2, col=1
                )
                
                fig.update_layout(height=800, showlegend=True,
                                title_text="Optimization Progress")
                fig.write_html(f"{self.save_dir}/optimization_history.html")
                
            # Try Optuna's built-in visualizations
            try:
                import optuna.visualization as vis
                
                # Optimization history
                fig = vis.plot_optimization_history(self.study)
                fig.write_html(f"{self.save_dir}/optuna_optimization_history.html")
                
                # Parameter importances
                if len(completed_trials) > 10:
                    fig = vis.plot_param_importances(self.study)
                    fig.write_html(f"{self.save_dir}/param_importances.html")
                
                # Slice plot
                fig = vis.plot_slice(self.study)
                fig.write_html(f"{self.save_dir}/slice_plot.html")
                
                # Parallel coordinate
                if len(completed_trials) > 3:
                    fig = vis.plot_parallel_coordinate(self.study)
                    fig.write_html(f"{self.save_dir}/parallel_coordinate.html")
                
            except Exception as e:
                print(f"Some Optuna visualizations failed: {e}")
                
        except ImportError:
            print("Install plotly for visualizations: pip install plotly kaleido")
        except Exception as e:
            print(f"Visualization error: {e}")

def train_with_best_params(best_params, args, dataset, essays, y, embedded_df, save_dir):
    """최적 파라미터로 최종 모델 학습"""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*60)
    
    data = get_prepared_data(args, dataset, essays, y, embedded_df)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_outputs = data['y_train'].shape[1]
    
    # Create DataLoaders with best batch_size
    train_loader = DataLoader(
        EssayDataset(data['train_embs'], data['train_features'], data['y_train']), 
        batch_size=best_params['batch_size'], shuffle=True
    )
    valid_loader = DataLoader(
        EssayDataset(data['valid_embs'], data['valid_features'], data['y_valid']), 
        batch_size=best_params['batch_size'], shuffle=False
    )
    
    # Model with best parameters
    model = GRUScoreModuleWithLNUKTAAttention(
        n_outputs, 
        best_params['hidden_dim'], 
        ukt_a_dim=294, 
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # Training
    best_kappa = -1.0
    best_outputs = None
    early_stop = 0
    train_losses = []
    val_losses = []
    kappa_scores = []
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(1, 200):  # More epochs for final training
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
        with torch.no_grad():
            for xb, ub, yb in valid_loader:
                xb, ub, yb = xb.to(device), ub.to(device), yb.to(device)
                out, _ = model(xb, ub)
                val_loss += criterion(out, yb).item()
                outputs.append(out.cpu().numpy())
        
        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate metrics
        y_pred = np.vstack(outputs) * args["num_range"]
        y_true = np.array(data['y_valid']) * args["num_range"]
        metrics = compute_metrics(y_pred, y_true)
        current_kappa = metrics['average_kappa']
        kappa_scores.append(current_kappa)
        
        # Print progress
        if epoch % 10 == 0 or current_kappa > best_kappa:
            status = " ⭐ NEW BEST!" if current_kappa > best_kappa else ""
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Kappa: {current_kappa:.4f}{status}")
        
        if current_kappa > best_kappa:
            best_kappa = current_kappa
            best_outputs = np.vstack(outputs)
            best_metrics = metrics
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= best_params['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Final evaluation and save
    y_pred_final = best_outputs * args["num_range"]
    y_true_final = np.array(data['y_valid']) * args["num_range"]
    
    np.save(f"{save_dir}/y_pred_final.npy", y_pred_final)
    np.save(f"{save_dir}/y_true_final.npy", y_true_final)
    
    # Save detailed results
    final_results = {
        'best_kappa': float(best_kappa),
        'best_epoch': best_epoch,
        'individual_kappas': [float(k) for k in best_metrics['individual_kappas']],
        'rubric_names': rubric_names,
        'final_epoch': epoch,
        'hyperparameters': best_params
    }
    
    with open(f"{save_dir}/final_model_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'kappa_score': kappa_scores
    })
    history_df.to_csv(f"{save_dir}/training_history.csv", index=False)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL MODEL RESULTS")
    print("="*60)
    print(f"Best Average Kappa: {best_kappa:.6f} (at epoch {best_epoch})")
    print(f"\nIndividual Kappa Scores:")
    print("-" * 60)
    for i, (name, kappa) in enumerate(zip(rubric_names, best_metrics['individual_kappas'])):
        if i != 9:  # Skip cont-prompt
            print(f"  {name:25s}: {kappa:.4f}")
    print("="*60)
    
    return best_kappa, best_metrics

def main():
    """Main function for Optuna hyperparameter search"""
    
    # Configuration  
    config["mode"] = "gru_with_ln_ukta_attention"
    config["is_topic_label"] = True  # As requested
    
    args = config["aihub_v1"]
    
    print("="*60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("Model: GRU with LayerNorm and UKTA Attention")
    print("="*60)
    print("\nLoading data...")
    
    # Load data
    DATASET = pd.read_csv(args["dataset_path"], encoding="utf-8-sig")
    ESSAYS, Y = get_essay_dataset_11_rubrics(is_rubric=True, args=args)
    
    emb_file = os.path.join(
        args["emb_file_path"],
        f"{args['dataset_path'].split('/')[1]}_{'notlabeled' if not config['is_topic_label'] else 'labeled'}.csv",
    )
    print(f"Loading embeddings from: {emb_file}")
    EMBEDDED_DF = dd.read_csv(emb_file, encoding="cp949", header=None).compute()
    
    print(f"Data loaded successfully!")
    print(f"  - Essays: {len(ESSAYS)}")
    print(f"  - Features: {len(features)}")
    print(f"  - Embeddings shape: {EMBEDDED_DF.shape}")
    
    # Create searcher and run optimization
    searcher = OptunaHyperparameterSearcher(
        n_trials=100,  # Adjust based on your computational budget
        study_name="gru_ukta_attention_optimization"
    )
    
    # Run hyperparameter search
    searcher.search(
        args=args,
        dataset=DATASET,
        essays=ESSAYS,
        y=Y,
        embedded_df=EMBEDDED_DF
    )
    
    # Train final model with best parameters
    if searcher.study and searcher.study.best_params:
        best_params = searcher.study.best_params
        train_with_best_params(
            best_params, args, DATASET, ESSAYS, Y, EMBEDDED_DF, searcher.save_dir
        )
        
        print(f"\n{'='*60}")
        print(f"All results saved to: {searcher.save_dir}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()