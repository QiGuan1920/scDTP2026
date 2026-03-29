import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import euclidean

# ==================== Evaluation Functions ====================
def find_top_differential_genes(expr_data, control_data, top_n=100):
    """Find differentially expressed genes."""
    expr_mean = expr_data.mean(axis=0)
    control_mean = control_data.mean(axis=0)
    fold_change = np.abs(expr_mean - control_mean) / (control_mean + 1e-8)
    top_indices = np.argsort(fold_change)[-top_n:]
    return top_indices, fold_change[top_indices]

def calculate_deg_identification_rate(real_expr, pred_expr, control_expr, top_n=100):
    """Calculate DEG identification rate."""
    real_expr_np = real_expr.cpu().numpy() if torch.is_tensor(real_expr) else real_expr
    pred_expr_np = pred_expr.cpu().numpy() if torch.is_tensor(pred_expr) else pred_expr
    control_expr_np = control_expr.cpu().numpy() if torch.is_tensor(control_expr) else control_expr
    
    real_deg_indices, _ = find_top_differential_genes(real_expr_np, control_expr_np, top_n)
    pred_deg_indices, _ = find_top_differential_genes(pred_expr_np, control_expr_np, top_n)
    
    overlap = len(set(real_deg_indices) & set(pred_deg_indices))
    return overlap / top_n, real_deg_indices, pred_deg_indices

def evaluate_model_comprehensive(model, control_expr, drug_features, real_drug_expr, device, control_data_for_deg=None):
    """Comprehensively evaluate model performance."""
    model.eval()
    with torch.no_grad():
        control_expr = control_expr.to(device)
        drug_features = {k: v.to(device) for k, v in drug_features.items()}
        real_drug_expr = real_drug_expr.to(device)
        
        pred_expr = model(control_expr, drug_features, mode='generate')
        
        pred_gene_mean = pred_expr.mean(dim=0).cpu().numpy()
        real_gene_mean = real_drug_expr.mean(dim=0).cpu().numpy()
        
        pred_gene_median = torch.median(pred_expr, dim=0)[0].cpu().numpy()
        real_gene_median = torch.median(real_drug_expr, dim=0)[0].cpu().numpy()
        
        # Mean metrics
        r2_mean = r2_score(real_gene_mean, pred_gene_mean)
        pearson_r_mean, pearson_p_mean = pearsonr(real_gene_mean, pred_gene_mean)
        spearman_r_mean, spearman_p_mean = spearmanr(real_gene_mean, pred_gene_mean)
        mse_mean = mean_squared_error(real_gene_mean, pred_gene_mean)
        rmse_mean = np.sqrt(mse_mean)
        
        # Median metrics
        r2_median = r2_score(real_gene_median, pred_gene_median)
        pearson_r_median, pearson_p_median = pearsonr(real_gene_median, pred_gene_median)
        spearman_r_median, spearman_p_median = spearmanr(real_gene_median, pred_gene_median)
        mse_median = mean_squared_error(real_gene_median, pred_gene_median)
        rmse_median = np.sqrt(mse_median)
        
        # Distribution distance
        pred_center = pred_expr.mean(dim=0).cpu().numpy()
        real_center = real_drug_expr.mean(dim=0).cpu().numpy()
        euclidean_dist = euclidean(pred_center, real_center)
        
        # DEG Identification Rate
        deg_rate_100 = 0
        if control_data_for_deg is not None:
            deg_rate_100, _, _ = calculate_deg_identification_rate(
                real_drug_expr.cpu(), pred_expr.cpu(), control_data_for_deg, top_n=100
            )
        
        # Log likelihood
        try:
            condition = model.encode_features(control_expr, drug_features)
            log_prob = model.flow_model.log_prob(real_drug_expr, condition)
            log_likelihood = log_prob.mean().item()
        except:
            log_likelihood = float('nan')
        
        return {
            'r2_mean': r2_mean, 'pearson_r_mean': pearson_r_mean, 'pearson_p_mean': pearson_p_mean,
            'spearman_r_mean': spearman_r_mean, 'spearman_p_mean': spearman_p_mean,
            'mse_mean': mse_mean, 'rmse_mean': rmse_mean,
            'r2_median': r2_median, 'pearson_r_median': pearson_r_median, 'pearson_p_median': pearson_p_median,
            'spearman_r_median': spearman_r_median, 'spearman_p_median': spearman_p_median,
            'mse_median': mse_median, 'rmse_median': rmse_median,
            'euclidean_distance': euclidean_dist,
            'deg_identification_rate_100': deg_rate_100,
            'log_likelihood': log_likelihood,
            'n_pred_cells': pred_expr.shape[0],
            'n_real_cells': real_drug_expr.shape[0]
        }

def export_perturbation_adata(adata, target_cell_type, target_drug, control_all, 
                              real_expr, pred_expr, save_path=None):
    """Export predictions to AnnData"""
    control_np = control_all.detach().cpu().numpy()
    pred_np = pred_expr.detach().cpu().numpy()

    blocks = [control_np]
    groups = ['Control'] * control_np.shape[0]
    conditions = ['Control'] * control_np.shape[0]
    obs_names = [f"ctrl_{i}" for i in range(control_np.shape[0])]

    if real_expr is not None:
        real_np = real_expr.detach().cpu().numpy()
        blocks.append(real_np)
        groups += ['Real'] * real_np.shape[0]
        conditions += [target_drug] * real_np.shape[0]
        obs_names += [f"real_{i}" for i in range(real_np.shape[0])]

    blocks.append(pred_np)
    groups += ['Pred'] * pred_np.shape[0]
    conditions += [target_drug] * pred_np.shape[0]
    obs_names += [f"pred_{i}" for i in range(pred_np.shape[0])]

    X = np.vstack(blocks)

    obs_df = pd.DataFrame({
        'group': groups,
        'condition': conditions,
        'cell_type': [target_cell_type] * X.shape[0],
    }, index=obs_names)

    export_adata = ad.AnnData(X=X, obs=obs_df, var=adata.var.copy())

    if save_path:
        export_adata.write_h5ad(save_path)
        print(f"✓ Saved AnnData to: {save_path}")

    return export_adata

def get_eval_metrics(
    eval_adata,
    key_dic,
    n_degs: int = 100,
    rank_method: str = "wilcoxon",
):
    """
    Calculate multiple groups of evaluation metrics (All genes & DEG-based; Mean & Median-based):
      - R2 / RMSE / MSE / Pearson / Spearman / Euclidean / Cosine
      - DEG identification rate (real vs pred, top n_degs)

    Parameters
    ----
    eval_adata : AnnData
        obs must contain condition information (e.g., "condition"), with ctrl / stim / pred categories.
    key_dic : dict
        Must contain keys:
          - 'condition_key': condition column name in obs
          - 'ctrl_key' / 'stim_key' / 'pred_key': values for the three conditions
    n_degs : int, default=100
        Take the top N DEGs for DEG identification rate and "DEG-based metrics".
    rank_method : str, default="wilcoxon"
        Differential analysis method for scanpy.tl.rank_genes_groups.

    Returns
    ----
    metrics_long : pd.DataFrame ("Long format table")
        Columns include:
          ['scope','stat','measure','value','better']
        Where:
          - scope ∈ {'all','degs'} (All genes / DEG subset)
          - stat  ∈ {'mean','median','top_n'} (Mean / Median / Special: for DEG rate)
          - measure is the specific metric name:
              {'r2','rmse','mse','pearson','spearman','euclidean','cosine_sim','deg_identification_rate'}
          - better ∈ {'higher','lower'}
    """

    # ---------- Read keys ----------
    condition_key = key_dic['condition_key']
    ctrl_key      = key_dic['ctrl_key']
    stim_key      = key_dic['stim_key']
    pred_key      = key_dic['pred_key']

    # ---------- Differential analysis: relative to ctrl ----------
    sc.tl.rank_genes_groups(
        eval_adata,
        groupby=condition_key,
        reference=ctrl_key,
        method=rank_method
    )
    # Top n DEGs for true stim
    degs_real = list(eval_adata.uns["rank_genes_groups"]["names"][stim_key][:n_degs])
    # Top n DEGs for pred (used for identification rate)
    degs_pred = list(eval_adata.uns["rank_genes_groups"]["names"][pred_key][:n_degs])

    # ---------- Extract expression matrices ----------
    df_stim = eval_adata[eval_adata.obs[condition_key] == stim_key].to_df()
    df_pred = eval_adata[eval_adata.obs[condition_key] == pred_key].to_df()

    # Align gene columns to avoid KeyError / inconsistency
    common_genes = df_stim.columns.intersection(df_pred.columns)
    df_stim = df_stim[common_genes]
    df_pred = df_pred[common_genes]

    # ---------- Basic stats: Mean & Median ----------
    def col_mean(df, genes=None):
        sub = df if genes is None else df[genes]
        return sub.mean(axis=0).values

    def col_median(df, genes=None):
        sub = df if genes is None else df[genes]
        return sub.median(axis=0).values

    # ---------- Metric calculation for two vectors ----------
    def safe_pearson(x, y):
        try:
            if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
                return np.nan
            return pearsonr(x, y)[0]
        except Exception:
            return np.nan

    def safe_spearman(x, y):
        try:
            return spearmanr(x, y).correlation
        except Exception:
            return np.nan

    def cosine_sim(x, y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx == 0 or ny == 0:
            return np.nan
        return float(np.dot(x, y) / (nx * ny))

    def vec_metrics(x, y):
        mse = mean_squared_error(x, y)
        rmse = np.sqrt(mse)
        try:
            r2 = r2_score(x, y)
        except Exception:
            r2 = np.nan
        p  = safe_pearson(x, y)
        s  = safe_spearman(x, y)
        eu = float(np.linalg.norm(x - y))
        cs = cosine_sim(x, y)
        return {
            'r2': r2,
            'rmse': rmse,
            'mse': mse,
            'pearson': p,
            'spearman': s,
            'euclidean': eu,
            'cosine_sim': cs,
        }

    # ---------- All genes & DEG subsets ----------
    mean_all_stim   = col_mean(df_stim)
    mean_all_pred   = col_mean(df_pred)
    median_all_stim = col_median(df_stim)
    median_all_pred = col_median(df_pred)

    degs_real_in_common = [g for g in degs_real if g in common_genes]
    if len(degs_real_in_common) == 0:
        degs_real_in_common = list(common_genes)

    mean_degs_stim   = col_mean(df_stim,   degs_real_in_common)
    mean_degs_pred   = col_mean(df_pred,   degs_real_in_common)
    median_degs_stim = col_median(df_stim, degs_real_in_common)
    median_degs_pred = col_median(df_pred, degs_real_in_common)

    # ---------- Assemble long table ----------
    better_rule = {
        'r2': 'higher',
        'rmse': 'lower',
        'mse': 'lower',
        'pearson': 'higher',
        'spearman': 'higher',
        'euclidean': 'lower',
        'cosine_sim': 'higher',
        'deg_identification_rate': 'higher',
    }

    records = []

    # All genes - Mean
    for k, v in vec_metrics(mean_all_stim, mean_all_pred).items():
        records.append({'scope':'all','stat':'mean','measure':k,'value':v,'better':better_rule[k]})

    # All genes - Median
    for k, v in vec_metrics(median_all_stim, median_all_pred).items():
        records.append({'scope':'all','stat':'median','measure':k,'value':v,'better':better_rule[k]})

    # DEGs - Mean
    for k, v in vec_metrics(mean_degs_stim, mean_degs_pred).items():
        records.append({'scope':'degs','stat':'mean','measure':k,'value':v,'better':better_rule[k]})

    # DEGs - Median
    for k, v in vec_metrics(median_degs_stim, median_degs_pred).items():
        records.append({'scope':'degs','stat':'median','measure':k,'value':v,'better':better_rule[k]})

    # DEG Identification rate
    overlap_rate = len(set(degs_real) & set(degs_pred)) / max(1, n_degs)
    records.append({
        'scope':'degs',
        'stat':'top_n',
        'measure':'deg_identification_rate',
        'value': overlap_rate,
        'better': better_rule['deg_identification_rate']
    })

    metrics_long = pd.DataFrame.from_records(records, columns=['scope','stat','measure','value','better'])
    return metrics_long
