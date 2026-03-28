import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from .utils import set_all_seeds

def detect_data_scenario(adata):
    """Detect the data scenario type."""
    treatment_data = adata[adata.obs['condition'] != 'Control']
    unique_cells = treatment_data.obs['cell_type'].nunique()
    unique_drugs = treatment_data.obs['condition'].nunique()
    
    print(f"Detected: {unique_cells} cell types, {unique_drugs} drugs")
    
    if unique_drugs == 1 and unique_cells > 1: return 'mono_drug_multi_cell'
    elif unique_drugs > 1 and unique_cells > 1: return 'multi_drug_multi_cell'
    elif unique_drugs > 1 and unique_cells == 1: return 'multi_drug_mono_cell'
    return 'single_combination'

def prepare_data_for_target(adata, drug_attr, drug_sm, drug_img, drug_net, 
                           target_cell_type=None, target_drug=None, scenario_type=None, seed=1000):
    """Prepare train, valid, and test data for a specific target."""
    set_all_seeds(seed)
    if scenario_type is None: scenario_type = detect_data_scenario(adata)
    
    treatment_data = adata[adata.obs['condition'] != 'Control']
    all_combinations = [
        (cell, drug) for cell in treatment_data.obs['cell_type'].unique()
        for drug in treatment_data.obs['condition'].unique()
        if ((adata.obs['cell_type'] == cell) & (adata.obs['condition'] == drug)).sum() > 0
        and drug in drug_attr
    ]
    
    # Determine test targets
    if scenario_type == 'mono_drug_multi_cell':
        target_cell_type = target_cell_type or all_combinations[0][0]
        target_drug = all_combinations[0][1]
        other_combinations = [(c, d) for c, d in all_combinations if c != target_cell_type]
    elif scenario_type == 'multi_drug_mono_cell':
        target_drug = target_drug or all_combinations[0][1]
        target_cell_type = all_combinations[0][0]
        other_combinations = [(c, d) for c, d in all_combinations if d != target_drug]
    elif scenario_type == 'multi_drug_multi_cell':
        target_cell_type = target_cell_type or all_combinations[0][0]
        target_drug = target_drug or all_combinations[0][1]
        other_combinations = [(c, d) for c, d in all_combinations if (c, d) != (target_cell_type, target_drug)]
    
    print(f"Scenario: {scenario_type} | Test Target: {target_cell_type} × {target_drug}")
    
    random.seed(seed)
    random.shuffle(other_combinations)
    
    if len(other_combinations) >= 3:
        train_combinations = other_combinations[:-1]
        valid_combinations = [other_combinations[-1]]
    elif len(other_combinations) == 2:
        train_combinations = other_combinations
        valid_combinations = [other_combinations[-1]]
    else:
        print("❌ Insufficient training data")
        return None
    
    def get_combination_data(combinations):
        all_control_expr, all_drug_features, all_drug_expr = [], [], []
        for cell_type, drug_name in combinations:
            control_mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == 'Control')
            drug_mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == drug_name)
            control_cells, drug_cells = adata[control_mask], adata[drug_mask]
            
            if control_cells.n_obs == 0 or drug_cells.n_obs == 0: continue
            
            control_expr = torch.tensor(control_cells.X, dtype=torch.float32)
            drug_expr = torch.tensor(drug_cells.X, dtype=torch.float32)
            
            drug_features = {
                'attr': drug_attr[drug_name], 'sm': drug_sm[drug_name],
                'img': drug_img[drug_name], 'net': drug_net[drug_name]
            }
            
            min_cells = min(control_expr.shape[0], drug_expr.shape[0])
            if min_cells > 0:
                generator = torch.Generator().manual_seed(seed + hash((cell_type, drug_name)) % 10000)
                control_indices = torch.randperm(control_expr.shape[0], generator=generator)[:min_cells]
                drug_indices = torch.randperm(drug_expr.shape[0], generator=generator)[:min_cells]
                
                paired_control = control_expr[control_indices]
                paired_drug = drug_expr[drug_indices]
                
                expanded_features = {key: feat.unsqueeze(0).repeat(min_cells, 1) for key, feat in drug_features.items()}
                
                all_control_expr.append(paired_control)
                all_drug_features.append(expanded_features)
                all_drug_expr.append(paired_drug)
        
        if all_control_expr:
            combined_control = torch.cat(all_control_expr, dim=0)
            combined_drug_expr = torch.cat(all_drug_expr, dim=0)
            combined_features = {key: torch.cat([df[key] for df in all_drug_features], dim=0) for key in all_drug_features[0].keys()}
            return (combined_control, combined_features, combined_drug_expr)
        return None
    
    train_data = get_combination_data(train_combinations)
    valid_data = get_combination_data(valid_combinations) 
    test_data = get_combination_data([(target_cell_type, target_drug)])
    test_control_mask = (adata.obs['cell_type'] == target_cell_type) & (adata.obs['condition'] == 'Control')
    control_all = torch.tensor(adata[test_control_mask].X, dtype=torch.float32)
    
    return {
        'train': train_data, 'valid': valid_data, 'test': test_data,
        'control_all': control_all, 'target_cell_type': target_cell_type,
        'target_drug': target_drug, 'scenario_type': scenario_type,
        'train_combinations': train_combinations, 'valid_combinations': valid_combinations
    }

def create_data_loader_unified(control_expr, drug_features, target_expr, batch_size=128, seed=1000):
    """Create a deterministic data loader."""
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        control_expr, drug_features['attr'], drug_features['sm'],
        drug_features['img'], drug_features['net'], target_expr
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator, num_workers=0)