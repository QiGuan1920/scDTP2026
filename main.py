import os
import torch
import pandas as pd
import numpy as np
from core.utils import get_device, set_all_seeds, test_reproducibility_cpu
from core.dataset import detect_data_scenario, prepare_data_for_target
from core.models import UnifiedDrugPredictor
from core.train import train_unified_model
from core.evaluate import evaluate_model_comprehensive, export_perturbation_adata, get_eval_metrics
from core.visualize import plot_training_curves, visualize_unified_results

device = get_device()

def test_single_target(adata, drug_attr, drug_sm, drug_img, drug_net, 
                      target_cell_type=None, target_drug=None, 
                      num_epochs=15, batch_size=128, lambda_recon=50.0, 
                      lambda_schedule=True, early_stopping=True, use_similarity=True, 
                      seed=1000, save_adata=False, save_adata_path=None, 
                      lambda_quantile=50.0, patience=10):
                      
    data_dict = prepare_data_for_target(adata, drug_attr, drug_sm, drug_img, drug_net, 
                                        target_cell_type, target_drug, seed=seed)
    if not data_dict or not data_dict['train']: return None
    
    train_control, train_drug_features, _ = data_dict['train']
    model = UnifiedDrugPredictor(
        gene_dim=train_control.shape[1],
        drug_attr_dim=train_drug_features['attr'].shape[1],
        drug_sm_dim=train_drug_features['sm'].shape[1],
        drug_img_dim=train_drug_features['img'].shape[1],
        drug_net_dim=train_drug_features['net'].shape[1],
        hidden_dim=256, seed=seed
    )
    if hasattr(model, 'use_similarity'): model.use_similarity = use_similarity
    
    train_losses, valid_losses = train_unified_model(
        model, data_dict, device, num_epochs=num_epochs, batch_size=batch_size,
        lambda_recon=lambda_recon, lambda_schedule=lambda_schedule, early_stopping=early_stopping,
        patience=patience, seed=seed, lambda_quantile=lambda_quantile,
        adata=adata, drug_attr=drug_attr, drug_sm=drug_sm, drug_img=drug_img, drug_net=drug_net
    )
    
    control_all = data_dict['control_all']
    test_drug_mask = (adata.obs['cell_type'] == data_dict['target_cell_type']) & (adata.obs['condition'] == data_dict['target_drug'])
    test_drug_cells = adata[test_drug_mask]
    real_test_expr = torch.tensor(test_drug_cells.X, dtype=torch.float32) if test_drug_cells.n_obs > 0 else None
    
    test_drug_features = {}
    for key, feat_dict in [('attr', drug_attr), ('sm', drug_sm), ('img', drug_img), ('net', drug_net)]:
        if data_dict['target_drug'] in feat_dict:
            test_drug_features[key] = feat_dict[data_dict['target_drug']].unsqueeze(0).repeat(control_all.shape[0], 1)
            
    test_metrics = evaluate_model_comprehensive(model, control_all, test_drug_features, 
                                               real_test_expr if real_test_expr is not None else control_all[:0, :], 
                                               device, control_data_for_deg=control_all)
    
    model_dev = model.to(device).eval()
    with torch.no_grad():
        pred_expr = model_dev(control_all.to(device), {k: v.to(device) for k, v in test_drug_features.items()}, mode='generate')
        
    export_adata = export_perturbation_adata(adata, data_dict['target_cell_type'], data_dict['target_drug'], 
                                            control_all, real_test_expr, pred_expr, save_path=save_adata_path)
    
    result_row = {
        'Target_Cell': data_dict['target_cell_type'], 'Target_Drug': data_dict['target_drug'],
        'Scenario': data_dict['scenario_type'], **test_metrics, 'Status': 'Success'
    }
    
    return {'target_cell_type': data_dict['target_cell_type'], 'target_drug': data_dict['target_drug'],
            'model': model, 'metrics': test_metrics, 'train_losses': train_losses, 
            'valid_losses': valid_losses, 'result_row': result_row, 'export_adata': export_adata}


def evaluate_all_targets(adata, drug_attr, drug_sm, drug_img, drug_net, 
                        num_epochs=15, batch_size=128, lambda_recon=50.0, 
                        lambda_schedule=True, early_stopping=True, use_similarity=True, 
                        seed=1000, lambda_quantile=50.0, patience=10, save_adata=False, 
                        save_adata_path=None, save_dir="./results/"):
    
    scenario_type = detect_data_scenario(adata)
    treatment_data = adata[adata.obs['condition'] != 'Control']
    
    test_targets = []
    if scenario_type == 'mono_drug_multi_cell':
        test_targets = [(cell, None) for cell in treatment_data.obs['cell_type'].unique()]
    elif scenario_type == 'multi_drug_mono_cell':
        test_targets = [(None, drug) for drug in treatment_data.obs['condition'].unique() if drug in drug_attr]
    elif scenario_type == 'multi_drug_multi_cell':
        for cell in treatment_data.obs['cell_type'].unique():
            for drug in treatment_data.obs['condition'].unique():
                if ((adata.obs['cell_type'] == cell) & (adata.obs['condition'] == drug)).sum() > 0 and drug in drug_attr:
                    test_targets.append((cell, drug))
    
    all_results, result_rows, all_X_long = [], [], []
    os.makedirs(save_dir, exist_ok=True)

    for i, (target_cell, target_drug) in enumerate(test_targets, 1):
        filename = f"prediction_{target_cell or 'mono_cell'}_{target_drug or 'mono_drug'}.h5ad"
        current_save_path = os.path.join(save_dir, filename) if save_adata else None
        
        result = test_single_target(
            adata, drug_attr, drug_sm, drug_img, drug_net, target_cell_type=target_cell, target_drug=target_drug,
            num_epochs=num_epochs, batch_size=batch_size, lambda_recon=lambda_recon, lambda_schedule=lambda_schedule,
            early_stopping=early_stopping, use_similarity=use_similarity, seed=seed,
            save_adata=save_adata, save_adata_path=current_save_path, lambda_quantile=lambda_quantile, patience=patience
        )
        
        if result:
            AA = result['export_adata']
            result['export_adata'] = None  # Reduce memory footprint
            all_results.append(result)
            result_rows.append(result['result_row'])
            
            # DEG and metrics evaluation
            AA.obs['condition'] = AA.obs['group'].replace({'Real': 'stimulated', 'Control': 'control', 'Pred': 'predict'})
            key_dic = {'condition_key': 'condition', 'ctrl_key': 'control', 'stim_key': 'stimulated', 'pred_key': 'predict'}
            
            try:
                X = get_eval_metrics(AA, key_dic, n_degs=100).copy()
                X['drug'] = target_cell if scenario_type != 'multi_drug_mono_cell' else target_drug
                all_X_long.append(X)
            except Exception as e:
                print(f"Failed to calculate evaluation metrics: {e}")
                
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            result_rows.append({'Target_Cell': target_cell, 'Target_Drug': target_drug, 'Status': 'Failed'})

    all_results.append(all_X_long)
    return all_results, pd.DataFrame(result_rows)


def main_unified_drug_prediction(adata, drug_attr, drug_sm, drug_img, drug_net, 
                                mode='evaluate_all', target_cell_type=None, target_drug=None,
                                num_epochs=15, batch_size=128, lambda_recon=50.0, lambda_quantile=50.0, 
                                early_stopping=True, patience=10, seed=1000, 
                                save_adata=True, save_dir="./results/"):
    
    print("=" * 80)
    print(f"Unified Drug Perturbation Prediction Model | Parameters: seed={seed}, epochs={num_epochs}, batch={batch_size}")
    print("=" * 80)
    
    if mode == 'evaluate_all':
        all_results, results_df = evaluate_all_targets(
            adata, drug_attr, drug_sm, drug_img, drug_net, num_epochs=num_epochs, batch_size=batch_size,
            lambda_recon=lambda_recon, early_stopping=early_stopping, seed=seed, lambda_quantile=lambda_quantile,
            save_adata=save_adata, save_dir=save_dir, patience=patience
        )
        visualize_unified_results(results_df, save_path=os.path.join(save_dir, "metrics_bar_plot.png") if save_dir else None)
        return all_results, results_df


# ==================== Execute Script ====================
if __name__ == "__main__":

    # Verify reproducibility in the test environment
    test_reproducibility_cpu(seed=1000)
    
    # Example run (using the latest parameters requested by the user)
    """
    all_results, _ = main_unified_drug_prediction(
        adata, drug_attr, drug_sm, drug_img, drug_net,
        mode='evaluate_all',
        num_epochs=15,
        batch_size=128,
        lambda_recon=50.0,
        lambda_quantile=50.0,
        early_stopping=True,
        patience=10,
        seed=1000,
        save_adata=True,
        save_dir="./results/h5ad_final/"
    )
    """