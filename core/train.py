import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from .utils import set_all_seeds
from .dataset import create_data_loader_unified

def distribution_quantile_loss(y_true, y_pred, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    Distribution-level quantile matching loss.
    
    Args:
        y_true: [n_cells_real, n_genes] Real perturbed expression matrix
        y_pred: [n_cells_pred, n_genes] Predicted expression matrix
        quantiles: Set of quantiles Q
        
    Returns:
        q_vec: [len(quantiles)] Loss for each quantile
        q_mean: Scalar, mean of all quantile losses
    """
    q_losses = []
    for q in quantiles:
        # For each gene, compute empirical quantile over the cell population dimension
        true_q = torch.quantile(y_true, q, dim=0)
        pred_q = torch.quantile(y_pred, q, dim=0)
        
        # Pinball loss: ρ_q(e) = max(q·e, (q-1)·e)
        e = true_q - pred_q
        pinball = torch.maximum(q * e, (q - 1.0) * e)
        q_losses.append(pinball.mean())
        
    q_vec = torch.stack(q_losses)
    q_mean = q_vec.mean()
    return q_vec, q_mean

def train_unified_model(model, data_dict, device, num_epochs=15, lr=1e-3, 
                       batch_size=128, lambda_recon=50.0, lambda_schedule=True, 
                       early_stopping=True, patience=10, verbose=True, seed=1000,
                       lambda_quantile=50.0, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
                       adata=None, drug_attr=None, drug_sm=None, drug_img=None, drug_net=None):
    """Unified model training function (Multi-Quantile Loss)."""
    set_all_seeds(seed)
    model = model.to(device)
    model.scenario_type = data_dict['scenario_type']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_control, train_drug_features, train_expr = data_dict['train']
    valid_control, valid_drug_features, valid_expr = data_dict['valid'] if data_dict['valid'] else (None, None, None)
    has_validation = data_dict['valid'] is not None

    # Construct training combinations for similarity computation
    if hasattr(model, 'use_similarity') and model.use_similarity:
        model.training_combinations = []
        if adata is not None and drug_attr is not None:
            max_combinations = min(len(data_dict['train_combinations']), 10)
            for cell_type, drug_name in data_dict['train_combinations'][:max_combinations]:
                control_mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == 'Control')
                control_cells = adata[control_mask]
                if control_cells.n_obs > 0 and drug_name in drug_attr:
                    control_expr = torch.tensor(control_cells.X, dtype=torch.float32)
                    df = {'attr': drug_attr[drug_name].clone(), 'sm': drug_sm[drug_name].clone(), 
                          'img': drug_img[drug_name].clone(), 'net': drug_net[drug_name].clone()}
                    for k, v in df.items():
                        df[k] = v.unsqueeze(0) if v.dim() == 0 else v.flatten() if v.dim() > 1 else v
                    model.training_combinations.append({'control_expr': control_expr, 'drug_features': df, 'cell_type': cell_type, 'drug_name': drug_name})

    train_loader = create_data_loader_unified(train_control, train_drug_features, train_expr, batch_size=batch_size, seed=seed)

    def get_dynamic_weights(epoch, total_epochs):
        """
        Early stage: Heavy on Recon/Quantile (learn general shape)
        Late stage: Heavy on NLL (refine distribution)
        """
        progress = epoch / total_epochs
        nll_weight = 0.1 + 0.2 * progress
        rw = lambda_recon * (1.0 - 0.2 * progress)
        lq = lambda_quantile * (1.0 - 0.2 * progress)
        return nll_weight, rw, lq

    def compute_r2_score(pred_expr, true_expr):
        return r2_score(true_expr.mean(dim=0).cpu().numpy(), pred_expr.mean(dim=0).cpu().numpy())

    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stop_triggered = False

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        epoch_train_logs = []
        nll_w, recon_w, lq_w = get_dynamic_weights(epoch, num_epochs)

        for batch_control, batch_attr, batch_sm, batch_img, batch_net, batch_target in train_loader:
            batch_control, batch_target = batch_control.to(device), batch_target.to(device)
            b_df = {'attr': batch_attr.to(device), 'sm': batch_sm.to(device), 'img': batch_img.to(device), 'net': batch_net.to(device)}

            optimizer.zero_grad()
            log_prob = model(batch_control, b_df, target_expr=batch_target, mode='train')
            nll_loss = -log_prob.mean()

            pred_expr = model(batch_control, b_df, mode='generate')
            recon_loss = F.mse_loss(pred_expr.mean(dim=0), batch_target.mean(dim=0))
            q_vec, q_mean = distribution_quantile_loss(batch_target, pred_expr, quantiles)

            total_loss = nll_w * nll_loss + recon_w * recon_loss + lq_w * q_mean
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_logs.append({'total': total_loss.item(), 'nll': nll_loss.item(), 'recon': recon_loss.item(), 'q_mean': q_mean.item()})

        avg_train = {k: np.mean([l[k] for l in epoch_train_logs]) for k in epoch_train_logs[0].keys()}
        
        model.eval()
        with torch.no_grad():
            tr_pred = model(train_control.to(device), {k: v.to(device) for k, v in train_drug_features.items()}, mode='generate')
            avg_train['r2'] = compute_r2_score(tr_pred, train_expr.to(device))
        train_losses.append(avg_train)

        if has_validation:
            with torch.no_grad():
                v_ctrl, v_expr = valid_control.to(device), valid_expr.to(device)
                v_df = {k: v.to(device) for k, v in valid_drug_features.items()}
                
                v_nll = -model(v_ctrl, v_df, target_expr=v_expr, mode='train').mean()
                v_pred = model(v_ctrl, v_df, mode='generate')
                v_recon = F.mse_loss(v_pred.mean(dim=0), v_expr.mean(dim=0))
                _, v_q_mean = distribution_quantile_loss(v_expr, v_pred, quantiles)
                
                curr_valid_loss = v_nll.item() * nll_w + v_recon.item() * recon_w + lq_w * v_q_mean.item()
                valid_losses.append({'total': curr_valid_loss, 'r2': compute_r2_score(v_pred, v_expr)})

                if early_stopping:
                    if curr_valid_loss < best_valid_loss:
                        best_valid_loss = curr_valid_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            early_stop_triggered = True
                            if best_model_state is not None: model.load_state_dict(best_model_state)
                            break
        scheduler.step()
        if early_stop_triggered: break

    return train_losses, valid_losses