import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, valid_losses, save_path=None):
    """Visualize training curves."""
    epochs = list(range(1, len(train_losses) + 1))
    tr_total = [d.get('total', np.nan) for d in train_losses]
    va_total = [d.get('total', np.nan) for d in valid_losses] if valid_losses else None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, tr_total, label='Train Loss', lw=2, color='blue')
    if va_total:
        ax.plot(epochs[:len(va_total)], va_total, label='Valid Loss', lw=2, ls='--', color='red')
    
    ax.set_title('Training Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_unified_results(results_df, save_path=None):
    """Visualize comprehensive test results."""
    successful_results = results_df[results_df['Status'] == 'Success']
    if len(successful_results) == 0: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"{row['Target_Cell']}_{row['Target_Drug']}"[:15] for _, row in successful_results.iterrows()]

    axes[0].bar(range(len(successful_results)), successful_results['R²_mean'], color='skyblue', alpha=0.7)
    axes[0].set_title('R² Score'); axes[0].set_xticks(range(len(successful_results)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')

    axes[1].bar(range(len(successful_results)), successful_results['DEG_ID_Rate_100'], color='purple', alpha=0.7)
    axes[1].set_title('DEG Identification Rate'); axes[1].set_xticks(range(len(successful_results)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()