import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LightweightCouplingLayer(nn.Module):
    """Lightweight Real NVP coupling layer."""
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super().__init__()
        self.split_dim = input_dim // 2
        
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.split_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
    
    def forward(self, x, condition, reverse=False):
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        scale = self.scale_net(torch.cat([x1, condition], dim=1))
        translate = self.translate_net(torch.cat([x1, condition], dim=1))
        
        if not reverse:
            x2 = x2 * torch.exp(scale) + translate
            log_det = scale.sum(dim=1)
        else:
            x2 = (x2 - translate) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)
            
        return torch.cat([x1, x2], dim=1), log_det


class LightweightConditionalFlow(nn.Module):
    """Lightweight Conditional Normalizing Flow."""
    def __init__(self, input_dim, condition_dim, num_layers=4, hidden_dim=128, seed=1000):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        self.coupling_layers = nn.ModuleList([
            LightweightCouplingLayer(input_dim, condition_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('permutations', torch.stack([
            torch.randperm(input_dim, generator=generator) for _ in range(num_layers)
        ]))
        
        self.register_parameter('base_loc', nn.Parameter(torch.zeros(input_dim)))
        self.register_parameter('base_scale', nn.Parameter(torch.ones(input_dim)))
    
    def _base_distribution_log_prob(self, z):
        return -0.5 * torch.sum(
            ((z - self.base_loc) / torch.exp(self.base_scale))**2 + 
            2 * self.base_scale + np.log(2 * np.pi), dim=1
        )
    
    def forward(self, x, condition, reverse=False):
        log_det_total = 0
        if not reverse:
            for i, coupling in enumerate(self.coupling_layers):
                x = x[:, self.permutations[i]]
                x, log_det = coupling(x, condition, reverse=False)
                log_det_total += log_det
        else:
            for i, coupling in reversed(list(enumerate(self.coupling_layers))):
                x, log_det = coupling(x, condition, reverse=True)
                x = x[:, torch.argsort(self.permutations[i])]
                log_det_total += log_det
        return x, log_det_total
    
    def log_prob(self, x, condition):
        z, log_det = self.forward(x, condition, reverse=False)
        return self._base_distribution_log_prob(z) + log_det
    
    def sample(self, condition, num_samples=None):
        if num_samples is None:
            num_samples = condition.shape[0]
        z = torch.randn(num_samples, self.input_dim).to(condition.device)
        z = z * torch.exp(self.base_scale) + self.base_loc
        x, _ = self.forward(z, condition, reverse=True)
        return x


class UnifiedDrugPredictor(nn.Module):
    """Unified Drug Perturbation Predictor - based on PCA similarity and conditional flow."""
    def __init__(self, gene_dim, drug_attr_dim, drug_sm_dim, drug_img_dim, drug_net_dim, hidden_dim=256, seed=1000):
        super().__init__()
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU()
        )

        self.drug_attr_encoder = nn.Sequential(nn.Linear(drug_attr_dim, hidden_dim//4), nn.ReLU())
        self.drug_sm_encoder = nn.Sequential(nn.Linear(drug_sm_dim, hidden_dim//4), nn.ReLU())
        self.drug_img_encoder = nn.Sequential(nn.Linear(drug_img_dim, hidden_dim//4), nn.ReLU())
        self.drug_net_encoder = nn.Sequential(nn.Linear(drug_net_dim, hidden_dim//4), nn.ReLU())

        self.drug_attention = nn.MultiheadAttention(embed_dim=hidden_dim//4, num_heads=2, dropout=0.1, batch_first=True)
        self.drug_projection = nn.Sequential(nn.Linear(hidden_dim//4, hidden_dim//2), nn.ReLU())
        self.norm_drug = nn.LayerNorm(hidden_dim//4)
        self.drug_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )

        self.response_generator = nn.Sequential(
            nn.Linear(100 + hidden_dim//2 + hidden_dim//2, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2)
        )

        self.fusion = nn.Sequential(nn.Linear(3 * (hidden_dim//2), hidden_dim), nn.ReLU())

        self.flow_model = LightweightConditionalFlow(
            input_dim=gene_dim, condition_dim=hidden_dim,
            num_layers=4, hidden_dim=64, seed=seed
        )

        self.training_combinations = []
        self.scenario_type = None

    def encode_drug_features(self, drug_features):
        """Encode and fuse four-modality drug features."""
        drug_attr_feat = self.drug_attr_encoder(drug_features['attr'])
        drug_sm_feat = self.drug_sm_encoder(drug_features['sm'])
        drug_img_feat = self.drug_img_encoder(drug_features['img'])
        drug_net_feat = self.drug_net_encoder(drug_features['net'])

        if len(drug_attr_feat.shape) > 1:
            batch_size = drug_attr_feat.shape[0]
            if len(drug_sm_feat.shape) == 1: drug_sm_feat = drug_sm_feat.unsqueeze(0).repeat(batch_size, 1)
            if len(drug_img_feat.shape) == 1: drug_img_feat = drug_img_feat.unsqueeze(0).repeat(batch_size, 1)
            if len(drug_net_feat.shape) == 1: drug_net_feat = drug_net_feat.unsqueeze(0).repeat(batch_size, 1)
            drug_modalities = torch.stack([drug_attr_feat, drug_sm_feat, drug_img_feat, drug_net_feat], dim=2).transpose(1,2)
        else:
            drug_modalities = torch.stack([drug_attr_feat, drug_sm_feat, drug_img_feat, drug_net_feat], dim=0).unsqueeze(0)
            batch_size = 1

        drug_attn_output, _ = self.drug_attention(query=drug_modalities, key=drug_modalities, value=drug_modalities)
        drug_features_norm = self.norm_drug(drug_modalities + drug_attn_output)
        drug_ff_output = self.drug_feed_forward(drug_features_norm)
        drug_features_final = self.norm_drug(drug_features_norm + drug_ff_output)

        drug_feat_aggregated = drug_features_final.mean(dim=1)
        drug_feat_projected = self.drug_projection(drug_feat_aggregated)
        if batch_size == 1:
            drug_feat_projected = drug_feat_projected.squeeze(0)
        return drug_feat_projected

    def generate_prototype(self, target_control, target_drug_features, exclude_combinations=None):
        """Generate prototype based on PCA cosine similarity."""
        device = next(self.parameters()).device
        out_dim = self.response_generator[-1].out_features

        if len(getattr(self, "training_combinations", [])) == 0:
            return torch.zeros(out_dim, device=device)

        tgt_ctrl = target_control.to(device).float()
        ref_ctrl_list, ref_drug_feats, ref_ctrl_encoded = [], [], []

        for combo in self.training_combinations:
            if exclude_combinations and combo in exclude_combinations: continue
            rc_expr = combo['control_expr'].to(device).float()
            ref_ctrl_list.append(rc_expr)

            rd = self.encode_drug_features(combo['drug_features']).to(device)
            if rd.ndim > 1: rd = rd.mean(0)
            re = self.gene_encoder(combo['control_expr'].mean(0).to(device))
            ref_drug_feats.append(rd)
            ref_ctrl_encoded.append(re)

        if len(ref_ctrl_list) == 0:
            return torch.zeros(out_dim, device=device)

        # PCA dimension reduction to 100 dimensions
        blocks = [tgt_ctrl] + ref_ctrl_list
        N_counts = [b.shape[0] for b in blocks]
        X = torch.cat(blocks, dim=0)
        X_mean = X.mean(dim=0, keepdim=True)
        Xc = X - X_mean

        k = min(100, Xc.shape[1], max(1, Xc.shape[0] - 1))
        U, S, V = torch.pca_lowrank(Xc, q=k)
        Vk = V[:, :k]
        Z = Xc @ Vk

        # Calculate mean vector of each group
        idx = 0
        z_means = []
        for n in N_counts:
            z_means.append(Z[idx:idx+n].mean(dim=0))
            idx += n
        z_target, z_refs = z_means[0], torch.stack(z_means[1:], dim=0)

        # Cell similarity
        cell_sims = (F.cosine_similarity(z_refs, z_target.unsqueeze(0), dim=1) + 1) / 2

        # Drug similarity
        target_drug_feat = self.encode_drug_features(target_drug_features).to(device)
        if target_drug_feat.ndim > 1: target_drug_feat = target_drug_feat.mean(0)
        ref_drug_mat = torch.stack(ref_drug_feats, dim=0)
        drug_sims = (F.cosine_similarity(ref_drug_mat, target_drug_feat.unsqueeze(0), dim=1) + 1) / 2

        # Scenario weighting
        if self.scenario_type == 'mono_drug_multi_cell': sims = cell_sims
        elif self.scenario_type == 'multi_drug_mono_cell': sims = drug_sims
        else: sims = 0.5 * (cell_sims + drug_sims)

        weights = F.softmax(sims / 1.0, dim=0)

        # Prototype generation
        response_patterns = []
        for rd, re, rc_expr in zip(ref_drug_feats, ref_ctrl_encoded, ref_ctrl_list):
            rc_pca = (rc_expr - X_mean) @ Vk
            rc_mean = rc_pca.mean(dim=0)
            rp = self.response_generator(torch.cat([rc_mean, rd, re], dim=0))
            response_patterns.append(rp)

        H = torch.stack(response_patterns, dim=0)
        h_proto = torch.sum(weights.unsqueeze(1) * H, dim=0)
        return h_proto

    def encode_features(self, gene_expr, drug_features):
        """Fuse gene, prototype, and drug features."""
        gene_feat = self.gene_encoder(gene_expr)
        drug_feat = self.encode_drug_features(drug_features)
        h_proto = self.generate_prototype(gene_expr, drug_features)

        if gene_feat.ndim > 1:
            B = gene_feat.shape[0]
            if h_proto.ndim == 1: h_proto = h_proto.unsqueeze(0).repeat(B, 1)
            if drug_feat.ndim == 1: drug_feat = drug_feat.unsqueeze(0).repeat(B, 1)
        else:
            if h_proto.ndim > 1: h_proto = h_proto.squeeze()
            if drug_feat.ndim > 1: drug_feat = drug_feat.squeeze()

        fused_feat = torch.cat([gene_feat, h_proto, drug_feat], dim=-1)
        return self.fusion(fused_feat)

    def forward(self, gene_expr, drug_features, target_expr=None, mode='generate'):
        condition = self.encode_features(gene_expr, drug_features)
        if mode == 'train' and target_expr is not None:
            return self.flow_model.log_prob(target_expr, condition)
        elif mode == 'generate':
            return self.flow_model.sample(condition)
        raise ValueError("mode must be 'train' or 'generate'")