import os
import torch
import pandas as pd
import numpy as np

# Import functions from core modules
from core.utils import get_device, set_all_seeds, test_reproducibility_cpu
from core.dataset import detect_data_scenario, prepare_data_for_target
from core.models import UnifiedDrugPredictor
from core.train import train_unified_model
from core.evaluate import evaluate_model_comprehensive, export_perturbation_adata, get_eval_metrics
from core.visualize import plot_training_curves, visualize_unified_results

device = get_device()

# ==================== Test Functions ====================
def test_single_target(adata, drug_attr, drug_sm, drug_img, drug_net, 
                      target_cell_type=None, target_drug=None, 
                      num_epochs=30, batch_size=128, lambda_recon=50.0, 
                      lambda_schedule=True, early_stopping=True, use_similarity=True, 
                      seed=1000, save_adata=False, save_adata_path=None, 
                      lambda_quantile=50.0, patience=10):
    """Test a specific target individually"""
    
    print(f"\n{'='*80}")
    print(f"Single Test - Cell: {target_cell_type}, Drug: {target_drug}")
    print(f"Parameters: epochs={num_epochs}, batch={batch_size}, seed={seed}")
    print(f"{'='*80}")
    
    data_dict = prepare_data_for_target(
        adata, drug_attr, drug_sm, drug_img, drug_net, 
        target_cell_type, target_drug, seed=seed
    )
    
    if data_dict is None or data_dict['train'] is None:
        print("❌ Data preparation failed")
        return None
        
    train_control, train_drug_features, train_expr = data_dict['train']
    
    gene_dim = train_control.shape[1]
    drug_attr_dim = train_drug_features['attr'].shape[1]
    drug_sm_dim = train_drug_features['sm'].shape[1]
    drug_img_dim = train_drug_features['img'].shape[1]
    drug_net_dim = train_drug_features['net'].shape[1]
    
    print(f"Dimensions: Genes={gene_dim}, Drugs=[{drug_attr_dim}, {drug_sm_dim}, {drug_img_dim}, {drug_net_dim}]")
    
    model = UnifiedDrugPredictor(
        gene_dim=gene_dim,
        drug_attr_dim=drug_attr_dim,
        drug_sm_dim=drug_sm_dim,
        drug_img_dim=drug_img_dim,
        drug_net_dim=drug_net_dim,
        hidden_dim=256,
        seed=seed
    )
    
    if hasattr(model, 'use_similarity'):
        model.use_similarity = use_similarity
    
    print(f"Start training...")
    train_losses, valid_losses = train_unified_model(
        model, data_dict, device, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        lr=1e-3,
        lambda_recon=lambda_recon,
        lambda_schedule=lambda_schedule,
        early_stopping=early_stopping,
        patience=patience,
        seed=seed,
        lambda_quantile=lambda_quantile,
        adata=adata,
        drug_attr=drug_attr,
        drug_sm=drug_sm, 
        drug_img=drug_img,
        drug_net=drug_net
    )
    
    print(f"\n{'='*80}")
    print(f"Start Testing Prediction")
    print(f"{'='*80}")
    
    control_all = data_dict['control_all']
    target_cell_type = data_dict['target_cell_type']
    target_drug = data_dict['target_drug']
    
    test_drug_mask = (adata.obs['cell_type'] == target_cell_type) & (adata.obs['condition'] == target_drug)
    test_drug_cells = adata[test_drug_mask]
    real_test_expr = torch.tensor(test_drug_cells.X, dtype=torch.float32) if test_drug_cells.n_obs > 0 else None
    
    print(f"Data: Control={control_all.shape[0]}, True Perturbation={0 if real_test_expr is None else real_test_expr.shape[0]}")
    
    n_control = control_all.shape[0]
    test_drug_features = {}
    for key, feat_dict in [('attr', drug_attr), ('sm', drug_sm), ('img', drug_img), ('net', drug_net)]:
        if target_drug in feat_dict:
            feat = feat_dict[target_drug]
            test_drug_features[key] = feat.unsqueeze(0).repeat(n_control, 1)
    
    test_metrics = evaluate_model_comprehensive(
        model, control_all, test_drug_features, 
        real_test_expr if real_test_expr is not None else control_all[:0, :],
        device, control_data_for_deg=control_all
    )
    
    model_dev = model.to(device).eval()
    with torch.no_grad():
        control_dev = control_all.to(device)
        test_feats_dev = {k: v.to(device) for k, v in test_drug_features.items()}
        pred_expr = model_dev(control_dev, test_feats_dev, mode='generate')
    
    export_adata = export_perturbation_adata(
        adata=adata,
        target_cell_type=target_cell_type,
        target_drug=target_drug,
        control_all=control_all,
        real_expr=real_test_expr,
        pred_expr=pred_expr,
        save_path=save_adata_path
    )
    
    print(f"\n{'='*80}")
    print(f"Test Results - {target_cell_type} × {target_drug}")
    print(f"{'='*80}")
    print(f"Based on Mean: R²={test_metrics['r2_mean']:.4f}, Pearson={test_metrics['pearson_r_mean']:.4f}, Spearman={test_metrics['spearman_r_mean']:.4f}")
    print(f"Based on Median: R²={test_metrics['r2_median']:.4f}, Pearson={test_metrics['pearson_r_median']:.4f}")
    print(f"Others: Euclidean Distance={test_metrics['euclidean_distance']:.4f}, DEG ID Rate={test_metrics['deg_identification_rate_100']:.4f}")
    print(f"{'='*80}")
    
    result_row = {
        'Target_Cell': target_cell_type,
        'Target_Drug': target_drug,
        'Scenario': data_dict['scenario_type'],
        'R²_mean': test_metrics['r2_mean'],
        'Pearson_R_mean': test_metrics['pearson_r_mean'],
        'Spearman_R_mean': test_metrics['spearman_r_mean'],
        'MSE_mean': test_metrics['mse_mean'],
        'RMSE_mean': test_metrics['rmse_mean'],
        'R²_median': test_metrics['r2_median'],
        'Pearson_R_median': test_metrics['pearson_r_median'],
        'Spearman_R_median': test_metrics['spearman_r_median'],
        'MSE_median': test_metrics['mse_median'],
        'RMSE_median': test_metrics['rmse_median'],
        'Euclidean_Distance': test_metrics['euclidean_distance'],
        'DEG_ID_Rate_100': test_metrics['deg_identification_rate_100'],
        'Log_Likelihood': test_metrics['log_likelihood'],
        'Status': 'Success'
    }
    
    return {
        'target_cell_type': target_cell_type,
        'target_drug': target_drug,
        'model': model,
        'metrics': test_metrics,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'result_row': result_row,
        'data_dict': data_dict,
        'export_adata': export_adata,
        'save_adata_path': save_adata_path
    }

def evaluate_all_targets(adata, drug_attr, drug_sm, drug_img, drug_net, 
                        num_epochs=30, batch_size=128, lambda_recon=50.0, 
                        lambda_schedule=True, early_stopping=True, use_similarity=True, 
                        seed=1000, lambda_quantile=50.0, patience=10, save_adata=False, 
                        save_adata_path=None, save_dir=None):
    """Test all targets sequentially"""
    
    scenario_type = detect_data_scenario(adata)
    print(f"Scenario: {scenario_type}, Seed: {seed}, Epochs: {num_epochs}")
    
    treatment_data = adata[adata.obs['condition'] != 'Control']
    
    test_targets = []
    if scenario_type == 'mono_drug_multi_cell':
        for cell_type in treatment_data.obs['cell_type'].unique():
            test_targets.append((cell_type, None))
    elif scenario_type == 'multi_drug_mono_cell':
        for drug_name in treatment_data.obs['condition'].unique():
            if drug_name in drug_attr:
                test_targets.append((None, drug_name))
    elif scenario_type == 'multi_drug_multi_cell':
        for cell_type in treatment_data.obs['cell_type'].unique():
            for drug_name in treatment_data.obs['condition'].unique():
                mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == drug_name)
                if mask.sum() > 0 and drug_name in drug_attr:
                    test_targets.append((cell_type, drug_name))
    
    print(f"Found {len(test_targets)} test targets")
    
    all_results = []
    result_rows = []
    all_X_long = []

    for i, (target_cell, target_drug) in enumerate(test_targets, 1):
        print(f"\n[{i}/{len(test_targets)}] Testing: {target_cell} × {target_drug}")
        
        if save_adata:
            if target_cell and target_drug:
                filename = f"prediction_{target_cell}_{target_drug}.h5ad"
            elif target_cell:
                filename = f"prediction_{target_cell}_mono_drug.h5ad"
            elif target_drug:
                filename = f"prediction_mono_cell_{target_drug}.h5ad"
            else:
                filename = f"prediction_{i}.h5ad"
            
            save_adata_path = os.path.join(save_dir, filename)
        
        result = test_single_target(
            adata, drug_attr, drug_sm, drug_img, drug_net,
            target_cell_type=target_cell,
            target_drug=target_drug,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_recon=lambda_recon,
            lambda_schedule=lambda_schedule,
            early_stopping=early_stopping,
            use_similarity=use_similarity,
            seed=seed,
            save_adata=save_adata,
            save_adata_path=save_adata_path,
            lambda_quantile=lambda_quantile,
            patience=patience
        )
        
        if result is not None:
            AA = result['export_adata']
            result['export_adata'] = None  # Free up space
            
            all_results.append(result)
            result_rows.append(result['result_row'])
            print(f"✓ Completed, saved to: {save_adata_path}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            AA.obs['condition'] = AA.obs['group'].replace({
                'Real': 'stimulated',
                'Control': 'control',
                'Pred': 'predict'
            })
            key_dic = {
                'condition_key': 'condition',
                'ctrl_key': 'control',
                'stim_key': 'stimulated',
                'pred_key': 'predict',
            }
            n_degs = 100
            
            X = get_eval_metrics(AA, key_dic, n_degs=n_degs).copy()
            
            if scenario_type == 'mono_drug_multi_cell':
                X['drug'] = target_cell
            elif scenario_type == 'multi_drug_mono_cell':
                X['drug'] = target_drug
            elif scenario_type == 'multi_drug_multi_cell':
                X['drug'] = target_cell

            all_X_long.append(X)
        else:
            result_rows.append({
                'Target_Cell': target_cell,
                'Target_Drug': target_drug,
                'Scenario': scenario_type,
                **{k: np.nan for k in ['R²_mean', 'Pearson_R_mean', 'Spearman_R_mean', 
                                       'MSE_mean', 'RMSE_mean', 'R²_median', 
                                       'Pearson_R_median', 'Spearman_R_median',
                                       'MSE_median', 'RMSE_median', 'Euclidean_Distance',
                                       'DEG_ID_Rate_100', 'Log_Likelihood']},
                'Status': 'Failed'
            })
            print(f"✗ Failed")

    all_results.append(all_X_long)
    final_results_df = pd.DataFrame(result_rows)
    
    print(f"\n{'='*120}")
    print(f"Sequential Testing Completed - Final Results")
    print(f"{'='*120}")
    
    successful_results = final_results_df[final_results_df['Status'] == 'Success']
    
    if len(successful_results) > 0:
        print(f"\nSuccess: {len(successful_results)}/{len(final_results_df)}")
        print(f"Average Metrics:")
        print(f"  R² (Mean): {successful_results['R²_mean'].mean():.4f} ± {successful_results['R²_mean'].std():.4f}")
        print(f"  Pearson: {successful_results['Pearson_R_mean'].mean():.4f} ± {successful_results['Pearson_R_mean'].std():.4f}")
        print(f"  DEG ID Rate: {successful_results['DEG_ID_Rate_100'].mean():.4f} ± {successful_results['DEG_ID_Rate_100'].std():.4f}")
    else:
        print(f"\n⚠️ All tests failed!")
    
    print(f"{'='*120}")
    
    return all_results, final_results_df

# ==================== Main Function ====================
def main_unified_drug_prediction(adata, drug_attr, drug_sm, drug_img, drug_net, 
                                mode='evaluate_all', target_cell_type=None, target_drug=None,
                                num_epochs=30, batch_size=128, lr=1e-3, 
                                lambda_recon=50.0, lambda_quantile=50.0, 
                                lambda_schedule=True, early_stopping=True, patience=10,
                                use_similarity=True, seed=1000, save_adata=False, 
                                save_adata_path=None, save_dir=None):
    """Unified drug perturbation prediction main function"""
    print("=" * 80)
    print("Unified Drug Perturbation Prediction Model")
    print("=" * 80)
    print(f"Parameters: seed={seed}, epochs={num_epochs}, batch={batch_size}")
    print(f"Loss weights: recon={lambda_recon}, quantile={lambda_quantile}")
    print(f"Early stop: {early_stopping} (patience={patience}), Similarity: {use_similarity}")
    print("=" * 80)
    
    if mode == 'evaluate_all':
        all_results, results_df = evaluate_all_targets(
            adata, drug_attr, drug_sm, drug_img, drug_net,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_recon=lambda_recon,
            lambda_schedule=lambda_schedule,
            early_stopping=early_stopping,
            use_similarity=use_similarity,
            seed=seed,
            lambda_quantile=lambda_quantile,
            save_adata=save_adata,
            save_adata_path=save_adata_path,
            save_dir=save_dir,
            patience=patience
        )
        
        print("\nGenerating visualizations...")
        visualize_unified_results(results_df, save_path=None)
        
        return all_results, results_df
        
    elif mode == 'single':
        result = test_single_target(
            adata, drug_attr, drug_sm, drug_img, drug_net,
            target_cell_type=target_cell_type,
            target_drug=target_drug,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_recon=lambda_recon,
            lambda_schedule=lambda_schedule,
            early_stopping=early_stopping,
            use_similarity=use_similarity,
            seed=seed,
            save_adata=save_adata,
            save_adata_path=save_adata_path,
            lambda_quantile=lambda_quantile,
            patience=patience
        )
        
        if result:
            print("\nGenerating training curves...")
            plot_training_curves(result['train_losses'], result['valid_losses'],
                                save_path=None)
        
        return result
    
    else:
        raise ValueError("mode must be 'evaluate_all' or 'single'")

# ==================== Execute Script ====================
if __name__ == "__main__":
    # Test reproducibility in CPU environment
    test_reproducibility_cpu(seed=1000)
  
    all_results, metrics_df = main_unified_drug_prediction(
        adata, drug_attr, drug_sm, drug_img, drug_net,
        mode='evaluate_all',
        num_epochs=30,
        batch_size=128,
        lambda_recon=50,
        lambda_quantile=50,
        early_stopping=True,
        patience=10,
        seed=1000,
        save_adata=True,
        save_adata_path=None, 
        save_dir="./results/h5ad_final/"
    )
    """
