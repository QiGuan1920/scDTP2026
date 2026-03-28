import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import euclidean

def find_top_differential_genes(expr_data, control_data, top_n=100):
    """Find differentially expressed genes (DEGs)."""
    expr_mean = expr_data.mean(axis=0)
    control_mean = control_data.mean(axis=0)
    fold_change = np.abs(expr_mean - control_mean) / (control_mean + 1e-8)
    top_indices = np.argsort(fold_change)[-top_n:]
    return top_indices, fold_change[top_indices]

def calculate_deg_identification_rate(real_expr, pred_expr, control_expr, top_n=100):
    """Calculate the DEG identification rate."""
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
        
        # Mean-based metrics
        r2_mean = r2_score(real_gene_mean, pred_gene_mean)
        pearson_r_mean, _ = pearsonr(real_gene_mean, pred_gene_mean)
        spearman_r_mean, _ = spearmanr(real_gene_mean, pred_gene_mean)
        mse_mean = mean_squared_error(real_gene_mean, pred_gene_mean)
        
        # Median-based metrics
        r2_median = r2_score(real_gene_median, pred_gene_median)
        pearson_r_median, _ = pearsonr(real_gene_median, pred_gene_median)
        
        euclidean_dist = euclidean(pred_expr.mean(dim=0).cpu().numpy(), real_drug_expr.mean(dim=0).cpu().numpy())
        
        deg_rate_100 = 0
        if control_data_for_deg is not None:
            deg_rate_100, _, _ = calculate_deg_identification_rate(real_drug_expr.cpu(), pred_expr.cpu(), control_data_for_deg, top_n=100)
        
        try:
            condition = model.encode_features(control_expr, drug_features)
            log_likelihood = model.flow_model.log_prob(real_drug_expr, condition).mean().item()
        except:
            log_likelihood = float('nan')
        
        return {
            'r2_mean': r2_mean, 'pearson_r_mean': pearson_r_mean, 'spearman_r_mean': spearman_r_mean,
            'mse_mean': mse_mean, 'rmse_mean': np.sqrt(mse_mean),
            'r2_median': r2_median, 'pearson_r_median': pearson_r_median,
            'euclidean_distance': euclidean_dist, 'deg_identification_rate_100': deg_rate_100,
            'log_likelihood': log_likelihood
        }

def export_perturbation_adata(adata, target_cell_type, target_drug, control_all, real_expr, pred_expr, save_path=None):
    """Export the prediction results as an AnnData object."""
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
    obs_df = pd.DataFrame({'group': groups, 'condition': conditions, 'cell_type': [target_cell_type] * X.shape[0]}, index=obs_names)
    export_adata = ad.AnnData(X=X, obs=obs_df, var=adata.var.copy())

    if save_path:
        export_adata.write_h5ad(save_path)
    return export_adata

def get_eval_metrics(eval_adata, key_dic, n_degs=100, rank_method="wilcoxon"):
    """
    Compute multiple groups of evaluation metrics (All genes & DEG-based; Mean & Median-based):
      - R2 / RMSE / MSE / Pearson / Spearman / Euclidean / Cosine
      - DEG identification rate (Real vs Pred, top n_degs)

    Parameters
    ----------
    eval_adata : AnnData
        obs contains condition info (e.g., "condition"), holding ctrl / stim / pred categories.
    key_dic : dict
        Required keys:
          - 'condition_key': condition column name in obs
          - 'ctrl_key' / 'stim_key' / 'pred_key': values corresponding to the three conditions
    n_degs : int, default=100
        Top N DEGs taken when computing DEG identification rate and "DEG-based metrics".
    rank_method : str, default="wilcoxon"
        Differential analysis method for scanpy.tl.rank_genes_groups.

    Returns
    -------
    metrics_long : pd.DataFrame ("Long format table")
        Columns include:
          ['scope','stat','measure','value','better']
        Where:
          - scope ∈ {'all','degs'}  (All genes / DEG subset)
          - stat  ∈ {'mean','median','top_n'}
          - measure is the specific metric name:
              {'r2','rmse','mse','pearson','spearman','euclidean','cosine_sim','deg_identification_rate'}
          - better ∈ {'higher','lower'} (Indicates if a higher or lower value is better)

    Notes
    -----
    - Pearson/Spearman will return NaN if a constant vector is encountered (protected).
    - Cosine similarity range is [-1,1], higher is more similar.
    - This function calls rank_genes_groups and writes to adata.uns["rank_genes_groups"].
    """
    # Read keys
    condition_key, ctrl_key = key_dic['condition_key'], key_dic['ctrl_key']
    stim_key, pred_key = key_dic['stim_key'], key_dic['pred_key']

    # Differential expression analysis: relative to ctrl
    sc.tl.rank_genes_groups(eval_adata, groupby=condition_key, reference=ctrl_key, method=rank_method)
    
    # Top n DEGs for true stim
    degs_real = list(eval_adata.uns["rank_genes_groups"]["names"][stim_key][:n_degs])
    # Top n DEGs for pred (used for identification rate)
    degs_pred = list(eval_adata.uns["rank_genes_groups"]["names"][pred_key][:n_degs])

    # Extract expression matrices
    df_stim = eval_adata[eval_adata.obs[condition_key] == stim_key].to_df()
    df_pred = eval_adata[eval_adata.obs[condition_key] == pred_key].to_df()
    
    # Align gene columns to avoid KeyError / inconsistent columns
    common_genes = df_stim.columns.intersection(df_pred.columns)
    df_stim, df_pred = df_stim[common_genes], df_pred[common_genes]

    def safe_metrics(x, y):
        mse = mean_squared_error(x, y)
        try: 
            r2 = r2_score(x, y)
        except: 
            r2 = np.nan
        
        # Pearson handles constant arrays by returning nan
        p = pearsonr(x, y)[0] if not (np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0)) else np.nan
        
        # Cosine similarity: dot(x,y) / (||x||*||y||)
        cs = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) if np.linalg.norm(x) > 0 and np.linalg.norm(y) > 0 else np.nan
        
        return {'r2': r2, 'rmse': np.sqrt(mse), 'mse': mse, 'pearson': p, 'euclidean': float(np.linalg.norm(x - y)), 'cosine_sim': cs}

    # All genes
    mean_all_stim, mean_all_pred = df_stim.mean(axis=0).values, df_pred.mean(axis=0).values
    
    # Subset of true stim DEGs
    degs_real_in_common = [g for g in degs_real if g in common_genes] or list(common_genes)
    mean_degs_stim, mean_degs_pred = df_stim[degs_real_in_common].mean(axis=0).values, df_pred[degs_real_in_common].mean(axis=0).values

    # Assemble long table
    records = []
    better_rule = {'r2': 'higher', 'rmse': 'lower', 'mse': 'lower', 'pearson': 'higher', 'euclidean': 'lower', 'cosine_sim': 'higher', 'deg_identification_rate': 'higher'}

    for k, v in safe_metrics(mean_all_stim, mean_all_pred).items(): records.append({'scope':'all', 'stat':'mean', 'measure':k, 'value':v, 'better':better_rule[k]})
    for k, v in safe_metrics(mean_degs_stim, mean_degs_pred).items(): records.append({'scope':'degs', 'stat':'mean', 'measure':k, 'value':v, 'better':better_rule[k]})

    # DEG identification rate
    records.append({'scope':'degs', 'stat':'top_n', 'measure':'deg_identification_rate', 'value': len(set(degs_real) & set(degs_pred)) / max(1, n_degs), 'better': 'higher'})
    
    return pd.DataFrame.from_records(records, columns=['scope','stat','measure','value','better'])