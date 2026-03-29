import matplotlib.pyplot as plt
import numpy as np

# ==================== Visualization Functions ====================
def plot_training_curves(train_losses, valid_losses, save_path=None):
    """Training curves visualization"""
    
    def pick_series(logs, candidates):
        for k in candidates:
            if len(logs) > 0 and k in logs[0]:
                return [d.get(k, np.nan) for d in logs]
        return None

    epochs = list(range(1, len(train_losses) + 1))

    tr_total = pick_series(train_losses, ['total'])
    tr_nll = pick_series(train_losses, ['nll', 'NLL', 'nll_loss'])
    tr_recon = pick_series(train_losses, ['reconstruction', 'recon'])
    tr_qtl = pick_series(train_losses, ['quantile', 'q_mean'])
    tr_r2 = pick_series(train_losses, ['r2', 'R2'])

    va_total = pick_series(valid_losses, ['total']) if valid_losses else None
    va_nll = pick_series(valid_losses, ['nll', 'NLL', 'nll_loss']) if valid_losses else None
    va_recon = pick_series(valid_losses, ['reconstruction', 'recon']) if valid_losses else None
    va_qtl = pick_series(valid_losses, ['quantile', 'q_mean']) if valid_losses else None
    va_r2 = pick_series(valid_losses, ['r2', 'R2']) if valid_losses else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.reshape(2, 3)

    # Total Loss
    ax = axes[0, 0]
    if tr_total: ax.plot(epochs, tr_total, label='Train', lw=2, color='blue')
    if va_total: ax.plot(list(range(1, len(va_total) + 1)), va_total, label='Valid', lw=2, ls='--', color='red')
    ax.set_title('Total Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # NLL
    ax = axes[0, 1]
    if tr_nll: ax.plot(epochs, tr_nll, label='Train', lw=2, color='blue')
    if va_nll: ax.plot(list(range(1, len(va_nll) + 1)), va_nll, label='Valid', lw=2, ls='--', color='red')
    ax.set_title('NLL'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # Recon + Quantile
    ax = axes[0, 2]
    if tr_recon: ax.plot(epochs, tr_recon, label='Train Recon', lw=2, color='purple')
    if va_recon: ax.plot(list(range(1, len(va_recon) + 1)), va_recon, label='Valid Recon', lw=2, ls='--', color='orchid')
    if tr_qtl: ax.plot(epochs, tr_qtl, label='Train Quantile', lw=2, color='darkgreen')
    if va_qtl: ax.plot(list(range(1, len(va_qtl) + 1)), va_qtl, label='Valid Quantile', lw=2, ls='--', color='limegreen')
    ax.set_title('Recon / Quantile'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # R2
    ax = axes[1, 0]
    if tr_r2: ax.plot(epochs, tr_r2, label='Train', lw=2, color='blue')
    if va_r2: ax.plot(list(range(1, len(va_r2) + 1)), va_r2, label='Valid', lw=2, ls='--', color='red')
    ax.set_title('R2 Score'); ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # NLL log scale
    ax = axes[1, 1]
    if tr_nll: ax.plot(epochs, tr_nll, label='Train', lw=2, color='blue')
    if va_nll: ax.plot(list(range(1, len(va_nll) + 1)), va_nll, label='Valid', lw=2, ls='--', color='red')
    ax.set_yscale('log'); ax.set_title('NLL (log-scale)'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # Quantile
    ax = axes[1, 2]
    if tr_qtl: ax.plot(epochs, tr_qtl, label='Train', lw=2, color='darkgreen')
    if va_qtl: ax.plot(list(range(1, len(va_qtl) + 1)), va_qtl, label='Valid', lw=2, ls='--', color='limegreen')
    ax.set_title('Quantile Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
    
    plt.show()

def visualize_unified_results(results_df, save_path=None):
    """Unified test results visualization"""
    successful_results = results_df[results_df['Status'] == 'Success']
    
    if len(successful_results) == 0:
        print("❌ No successful results to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    labels = [f"{row['Target_Cell']}_{row['Target_Drug']}"[:15] 
             for _, row in successful_results.iterrows()]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'brown']
    
    # R² 
    axes[0,0].bar(range(len(successful_results)), successful_results['R²_mean'], 
                  color=colors[0], alpha=0.7, label='Mean')
    axes[0,0].bar(range(len(successful_results)), successful_results['R²_median'], 
                  color=colors[1], alpha=0.5, label='Median')
    axes[0,0].set_title('R² Score'); axes[0,0].set_ylabel('R²')
    axes[0,0].set_xticks(range(len(successful_results)))
    axes[0,0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    # Pearson
    axes[0,1].bar(range(len(successful_results)), successful_results['Pearson_R_mean'], 
                  color=colors[2], alpha=0.7)
    axes[0,1].set_title('Pearson R'); axes[0,1].set_ylabel('Pearson R')
    axes[0,1].set_xticks(range(len(successful_results)))
    axes[0,1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Spearman
    axes[0,2].bar(range(len(successful_results)), successful_results['Spearman_R_mean'], 
                  color=colors[3], alpha=0.7)
    axes[0,2].set_title('Spearman R'); axes[0,2].set_ylabel('Spearman R')
    axes[0,2].set_xticks(range(len(successful_results)))
    axes[0,2].set_xticklabels(labels, rotation=45, ha='right')
    axes[0,2].grid(True, alpha=0.3)
    
    # DEG ID Rate
    axes[1,0].bar(range(len(successful_results)), successful_results['DEG_ID_Rate_100'], 
                  color=colors[4], alpha=0.7)
    axes[1,0].set_title('DEG Identification Rate'); axes[1,0].set_ylabel('Rate')
    axes[1,0].set_xticks(range(len(successful_results)))
    axes[1,0].set_xticklabels(labels, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # MSE
    axes[1,1].bar(range(len(successful_results)), successful_results['MSE_mean'], 
                  color=colors[5], alpha=0.7)
    axes[1,1].set_title('MSE'); axes[1,1].set_ylabel('MSE')
    axes[1,1].set_xticks(range(len(successful_results)))
    axes[1,1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1,1].grid(True, alpha=0.3)
    
    # Euclidean Distance
    axes[1,2].bar(range(len(successful_results)), successful_results['Euclidean_Distance'], 
                  color='cyan', alpha=0.7)
    axes[1,2].set_title('Euclidean Distance'); axes[1,2].set_ylabel('Distance')
    axes[1,2].set_xticks(range(len(successful_results)))
    axes[1,2].set_xticklabels(labels, rotation=45, ha='right')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results visualization saved to: {save_path}")
    
    plt.show()
