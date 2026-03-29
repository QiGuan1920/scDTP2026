import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from .utils import set_all_seeds
from .dataset import create_data_loader_unified

# ==================== Training Functions ====================
def train_unified_model(model, data_dict, device, num_epochs=30, lr=1e-3, 
                       batch_size=128, lambda_recon=50.0, lambda_schedule=True, 
                       early_stopping=True, patience=10, verbose=True, seed=1000,
                       lambda_quantile=50.0, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
                       adata=None, drug_attr=None, drug_sm=None, drug_img=None, drug_net=None):
    """Unified model training function (Multi-Quantile Loss)."""
    
    def distribution_quantile_loss(y_true, y_pred, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        Distribution-level quantile matching loss.
        
        Args:
            y_true:  [n_cells_real, n_genes] True perturbed expression matrix
            y_pred:  [n_cells_pred, n_genes] Predicted expression matrix
            quantiles: Quantile set Q
        
        Returns:
            q_vec:   [len(quantiles)] Loss for each quantile
            q_mean:  Scalar, mean of all quantile losses
        """
        q_losses = []
        
        for q in quantiles:
            # For each gene, compute empirical quantile over the cell population dimension
            true_q = torch.quantile(y_true, q, dim=0)   # [n_genes]
            pred_q = torch.quantile(y_pred, q, dim=0)   # [n_genes]
            
            # Pinball loss: ρ_q(e) = max(q·e, (q-1)·e)
            e = true_q - pred_q                          # [n_genes]
            pinball = torch.maximum(q * e, (q - 1.0) * e)  # [n_genes]
            
            # Average over all genes
            q_losses.append(pinball.mean())
        
        q_vec = torch.stack(q_losses)        # [|Q|]
        q_mean = q_vec.mean()                # Scalar: (1/|Q|) Σ_q L_q
        
        return q_vec, q_mean

    set_all_seeds(seed)
    model = model.to(device)
    model.scenario_type = data_dict['scenario_type']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    if data_dict['train'] is None:
        print("❌ Training data is empty")
        return [], []

    train_control, train_drug_features, train_expr = data_dict['train']
    valid_control, valid_drug_features, valid_expr = (None, None, None)
    has_validation = False
    
    if data_dict['valid'] is not None:
        valid_control, valid_drug_features, valid_expr = data_dict['valid']
        has_validation = True
        print(f"✓ Validation set size: {valid_control.shape[0]} samples")
    else:
        print("⚠️ No validation data, early stopping disabled")

    # Construct training combinations
    if hasattr(model, 'use_similarity') and model.use_similarity:
        model.training_combinations = []
        
        if adata is None or drug_attr is None:
            print("⚠️ Missing necessary parameters, skipping similarity setup")
            model.use_similarity = False
        else:
            print("Constructing training combinations for similarity computation...")
            max_combinations = min(len(data_dict['train_combinations']), 10)
            
            for i, (cell_type, drug_name) in enumerate(data_dict['train_combinations'][:max_combinations]):
                try:
                    control_mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == 'Control')
                    control_cells = adata[control_mask]
                    
                    if control_cells.n_obs == 0 or drug_name not in drug_attr:
                        continue
                    
                    control_expr = torch.tensor(control_cells.X, dtype=torch.float32)
                    drug_features = {
                        'attr': drug_attr[drug_name].clone().detach(),
                        'sm': drug_sm[drug_name].clone().detach(),
                        'img': drug_img[drug_name].clone().detach(),
                        'net': drug_net[drug_name].clone().detach()
                    }
                    
                    for key, feat in drug_features.items():
                        if feat.dim() == 0:
                            drug_features[key] = feat.unsqueeze(0)
                        elif feat.dim() > 1:
                            drug_features[key] = feat.flatten()
                    
                    model.training_combinations.append({
                        'control_expr': control_expr,
                        'drug_features': drug_features,
                        'cell_type': cell_type,
                        'drug_name': drug_name
                    })
                    print(f"  ✓ Added combination {i+1}: {cell_type} × {drug_name} ({control_expr.shape[0]} cells)")
                except Exception as e:
                    print(f"  ⚠️ Error processing combination {cell_type} × {drug_name}: {e}")
                    continue
            
            print(f"✓ Successfully constructed {len(model.training_combinations)} training combinations")
            
            if len(model.training_combinations) == 0:
                model.use_similarity = False

    train_loader = create_data_loader_unified(
        train_control, train_drug_features, train_expr, 
        batch_size=batch_size, seed=seed
    )

    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stop_triggered = False
    
    print(f"\nEarly stopping config: Enabled={early_stopping}, Has_Valid={has_validation}, Patience={patience}")
    
    def get_dynamic_weights(epoch, total_epochs):
        progress = epoch / total_epochs
        
        # Early: Heavy on Recon/Quantile (learn rough shape)
        # Late: Heavy on NLL (refine distribution)
        nll_weight = 0.1 + 0.2 * progress        # 0.1 → 0.3
        recon_weight = lambda_recon * (1.0 - 0.2 * progress)  
        l_quantile = lambda_quantile * (1.0 - 0.2 * progress) 

        return nll_weight, recon_weight, l_quantile

    def compute_r2_score(pred_expr, true_expr):
        pred_mean = pred_expr.mean(dim=0).cpu().numpy()
        true_mean = true_expr.mean(dim=0).cpu().numpy()
        return r2_score(true_mean, pred_mean)

    show_qs = list(quantiles)
    q_cols = " ".join([f"Q@{q:.2f}".ljust(9) for q in show_qs])
    print(f"\nStarting training (with Multi-Quantile Loss)...")
    print(f"{'='*168}")
    print(f"{'Epoch':<6} {'Tr_Tot':<10} {'Tr_NLL':<10} {'Tr_Recon':<10} {'Tr_QtlMean':<11} {q_cols} {'Tr_R2':<9} "
          f"{'Va_Tot':<10} {'Va_NLL':<10} {'Va_Recon':<10} {'Va_QtlMean':<11} {'Va_R2':<9} "
          f"{'NLL_W':<7} {'Recon_W':<8} {'Qλ':<6} {'LR':<10} {'EarlyStop':<15}")
    print(f"{'='*168}")

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_train_logs = []

        nll_weight, recon_weight, l_quantile = get_dynamic_weights(epoch, num_epochs)

        for batch_idx, (batch_control, batch_attr, batch_sm, batch_img, batch_net, batch_target) in enumerate(train_loader):
            batch_control = batch_control.to(device)
            batch_drug_features = {
                'attr': batch_attr.to(device),
                'sm': batch_sm.to(device),
                'img': batch_img.to(device),
                'net': batch_net.to(device)
            }
            batch_target = batch_target.to(device)

            optimizer.zero_grad()

            log_prob = model(batch_control, batch_drug_features, target_expr=batch_target, mode='train')
            nll_loss = -log_prob.mean()

            pred_expr = model(batch_control, batch_drug_features, mode='generate')
            recon_loss = F.mse_loss(pred_expr.mean(dim=0), batch_target.mean(dim=0))

            q_vec, q_mean = distribution_quantile_loss(batch_target, pred_expr, quantiles)  

            total_loss = nll_weight * nll_loss + recon_weight * recon_loss + l_quantile * q_mean

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            row = {
                'total': total_loss.item(),
                'nll': nll_loss.item(),
                'recon': recon_loss.item(),
                'q_mean': q_mean.item(),
                'nll_weight': nll_weight,
                'recon_weight': recon_weight
            }
            for i, q in enumerate(show_qs):
                row[f"q@{q:.2f}"] = q_vec[i].item()
            epoch_train_logs.append(row)

        def avg_key(key):
            return float(np.mean([l[key] for l in epoch_train_logs]))
        
        avg_train = {
            'total': avg_key('total'),
            'nll': avg_key('nll'),
            'recon': avg_key('recon'),
            'q_mean': avg_key('q_mean'),
            'nll_weight': epoch_train_logs[0]['nll_weight'],
            'recon_weight': epoch_train_logs[0]['recon_weight'],
            'lr': optimizer.param_groups[0]['lr']
        }
        for q in show_qs:
            avg_train[f"q@{q:.2f}"] = avg_key(f"q@{q:.2f}")

        model.eval()
        with torch.no_grad():
            train_control_dev = train_control.to(device)
            train_drug_features_dev = {k: v.to(device) for k, v in train_drug_features.items()}
            train_expr_dev = train_expr.to(device)
            train_pred_full = model(train_control_dev, train_drug_features_dev, mode='generate')
            train_r2 = compute_r2_score(train_pred_full, train_expr_dev)

        avg_train['r2'] = train_r2
        train_losses.append(avg_train)

        current_valid_loss = float('inf')
        valid_log = {'total': np.nan, 'nll': np.nan, 'recon': np.nan, 'q_mean': np.nan, 'r2': np.nan}
        early_stop_info = "N/A"
        
        if has_validation:
            model.eval()
            with torch.no_grad():
                valid_control_dev = valid_control.to(device)
                valid_drug_features_dev = {k: v.to(device) for k, v in valid_drug_features.items()}
                valid_expr_dev = valid_expr.to(device)

                valid_log_prob = model(valid_control_dev, valid_drug_features_dev,
                                       target_expr=valid_expr_dev, mode='train')
                v_nll = -valid_log_prob.mean()

                v_pred = model(valid_control_dev, valid_drug_features_dev, mode='generate')
                v_recon = F.mse_loss(v_pred.mean(dim=0), valid_expr_dev.mean(dim=0))
                v_q_vec, _ = distribution_quantile_loss(valid_expr_dev, v_pred, quantiles)   
                v_q_mean = v_q_vec.mean()

                v_r2 = compute_r2_score(v_pred, valid_expr_dev)

                current_valid_loss = (v_nll.item() * nll_weight +
                                      v_recon.item() * recon_weight +
                                      l_quantile * v_q_mean.item())

                valid_log = {
                    'total': current_valid_loss,
                    'nll': v_nll.item(),
                    'recon': v_recon.item(),
                    'q_mean': v_q_mean.item(),
                    'r2': v_r2
                }
                for i, q in enumerate(show_qs):
                    valid_log[f"q@{q:.2f}"] = v_q_vec[i].item()

            if early_stopping:
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    early_stop_info = f"↓{patience_counter}/{patience}"
                else:
                    patience_counter += 1
                    early_stop_info = f"↑{patience_counter}/{patience}"
                    
                if patience_counter >= patience:
                    early_stop_triggered = True
                    print(f"\n⚠️ Early stopping triggered! epoch {epoch+1}, validation loss={current_valid_loss:.6f}, best={best_valid_loss:.6f}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print(f"✓ Restored best model state")
                    break
        else:
            early_stop_info = "No Valid Data"

        valid_losses.append(valid_log)
        scheduler.step()

        if verbose:
            lr_now = optimizer.param_groups[0]['lr']
            tr_q_cols = " ".join([f"{avg_train[f'q@{q:.2f}']:<9.4f}" for q in show_qs])
            va_q_cols = " ".join([f"{valid_log.get(f'q@{q:.2f}', float('nan')):<9.4f}" for q in show_qs])
            print(f"{epoch+1:<6} {avg_train['total']:<10.4f} {avg_train['nll']:<10.4f} {avg_train['recon']:<10.4f} {avg_train['q_mean']:<11.4f} {tr_q_cols} {avg_train['r2']:<9.4f} "
                  f"{valid_log['total']:<10.4f} {valid_log['nll']:<10.4f} {valid_log['recon']:<10.4f} {valid_log['q_mean']:<11.4f} {valid_log['r2']:<9.4f} "
                  f"{nll_weight:<7.3f} {recon_weight:<8.3f} {l_quantile:<6.3f} {lr_now:<10.2e} {early_stop_info:<15}")

    print(f"{'='*168}")
    
    if early_stop_triggered:
        print(f"✅ Training completed via early stopping! epoch {epoch+1}/{num_epochs}, best validation loss={best_valid_loss:.6f}")
    else:
        print(f"✅ Training completed for all {num_epochs} epochs!")

    return train_losses, valid_losses
