import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. SMPL Geometry Wrapper (For Virtual Observation Loss)
# -----------------------------------------------------------------------------
class SMPLGeometryWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vertex_indices = {
            'L_Toe': [3218, 3219, 3220, 3227, 3238, 3240, 3250, 3253, 3262, 3263, 3275, 3278, 3293, 3295, 3305, 3306],
            'L_Heel': [3458, 3459, 3460, 3461, 3463, 3466, 3467, 3468],
            'R_Toe': [6615, 6620, 6622, 6630, 6635, 6639, 6650, 6654, 6659, 6664, 6674, 6677, 6693, 6695, 6705, 6706],
            'R_Heel': [6858, 6859, 6860, 6861, 6864, 6866, 6867, 6868]
        }
        
        # --- NEW: STATIC OFFSETS (For HumanML3D / hml_vec) ---
        # Vectors calculated from User's Script: [X, Y, Z]
        self.register_buffer('offset_l_toe',  torch.tensor([-0.00957836, -0.01430488,  0.05481965]))
        self.register_buffer('offset_l_heel', torch.tensor([-0.01765292, -0.00870803, -0.15520199]))
        self.register_buffer('offset_r_toe',  torch.tensor([ 0.0088405,  -0.01663555,  0.05654152]))
        self.register_buffer('offset_r_heel', torch.tensor([ 0.01887021, -0.00870501, -0.15856517]))

        # Ground plane setup
        self.register_buffer('ground_normal', torch.tensor([0., 1., 0.]))
        self.register_buffer('ground_point', torch.tensor([0., 0., 0.]))
        self.register_buffer('P_parallel', torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]))

    def get_landmarks(self, vertices):
        """ 
        vertices: [B, Frames, 6890, 3] 
        Returns dict of centroids [B, Frames, 3]
        """
        landmarks = {}
        for name, indices in self.vertex_indices.items():
            if not indices:
                raise ValueError(f"Vertex indices for {name} are empty! Please extract them via Blender.")
            
            idx = torch.tensor(indices, device=vertices.device, dtype=torch.long)
            # Average the vertices to find the stable centroid of the contact patch
            landmarks[name] = vertices[:, :, idx, :].mean(dim=2) 
        return landmarks

    def compute_heights(self, landmarks):
        """ Returns signed height (distance to ground) """
        heights = {}
        for name, pos in landmarks.items():
            # Dot product: (p - p0) . n
            h = ((pos - self.ground_point) * self.ground_normal).sum(dim=-1)
            heights[name] = h
        return heights

    def compute_tangential_velocities(self, landmarks, dt=1/30.0):
        """ Returns squared magnitude of tangential velocity """
        velocities = {}
        for name, pos in landmarks.items():
            # Finite difference: v = (p_{t+1} - p_t) / dt
            delta = (pos[:, 1:] - pos[:, :-1]) / dt
            # Project onto ground plane: v_par = v . P
            v_par = torch.matmul(delta, self.P_parallel.t())
            # Return squared magnitude ||v||^2
            velocities[name] = (v_par ** 2).sum(dim=-1)
        return velocities


# -----------------------------------------------------------------------------
# 2. ContactNet (1D-CNN) - Used for Prior and Posterior
# -----------------------------------------------------------------------------
class ContactNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_on_text=False):
        super().__init__()
        self.condition_on_text = condition_on_text
        
        # If conditioning on text (Prior mode), input grows by hidden_dim
        effective_input = input_dim + hidden_dim if condition_on_text else input_dim
        
        # 1D Conv Network to smooth over time (Kernel 5 = +/- 2 frames context)
        self.net = nn.Sequential(
            nn.Conv1d(effective_input, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 4 * 2, kernel_size=1) # 1x1 conv to project to 8 logits (L/R * 4 classes)
        )

    def forward(self, x_motion, y_emb=None):
        """
        x_motion: [Batch, Time, Dim]
        y_emb: [Batch, 1, Dim] (Optional)
        """
        # Prepare inputs
        if self.condition_on_text:
            if y_emb is None:
                raise ValueError("ContactNet initialized with condition_on_text=True but y_emb is None")
            # Expand text to sequence length: [B, 1, D] -> [B, T, D]
            y_repeated = y_emb.expand(-1, x_motion.shape[1], -1)
            x_in = torch.cat([x_motion, y_repeated], dim=-1)
        else:
            x_in = x_motion
            
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        x_in = x_in.permute(0, 2, 1)
        
        # Forward
        out = self.net(x_in)
        
        # Permute back: [B, D_out, T] -> [B, T, D_out]
        out = out.permute(0, 2, 1)
        
        # Reshape to [Batch, Time, 2_feet, 4_classes]
        return out.view(out.shape[0], out.shape[1], 2, 4)


# -----------------------------------------------------------------------------
# 3. Terminal AR Decoder Components
# -----------------------------------------------------------------------------
class MixtureOfGaussiansHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_mixtures=10):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
        self.fc_pi = nn.Linear(hidden_dim, num_mixtures)
        self.fc_mu = nn.Linear(hidden_dim, num_mixtures * output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_mixtures * output_dim)

    def forward(self, x):
        B, S, _ = x.shape
        pi_logits = self.fc_pi(x) # [B, S, K]
        mu = self.fc_mu(x).view(B, S, self.num_mixtures, self.output_dim)
        # Log-sigma prediction clamped for stability
        log_sigma = self.fc_sigma(x).view(B, S, self.num_mixtures, self.output_dim)
        sigma = torch.exp(torch.clamp(log_sigma, min=-10, max=2))
        return pi_logits, mu, sigma

    def log_prob(self, x_target, pi_logits, mu, sigma):
        """ Calculate Log-Likelihood for NLL Loss """
        # x_target: [B, S, D] -> [B, S, 1, D]
        x_target = x_target.unsqueeze(2)
        var = sigma ** 2
        
        # Gaussian log-prob: -0.5 * (log(2pi) + 2log(sigma) + (x-mu)^2/var)
        # Sum over feature dimension D (assuming diagonal covariance)
        log_prob = -0.5 * (torch.log(2 * np.pi * var) + (x_target - mu)**2 / var)
        log_prob = torch.sum(log_prob, dim=-1) # [B, S, K]
        
        # LogSumExp trick: log(sum(pi * N(...)))
        # = logsumexp( log_softmax(pi) + log_prob_component )
        weighted_log_prob = torch.log_softmax(pi_logits, dim=-1) + log_prob
        return torch.logsumexp(weighted_log_prob, dim=-1) # [B, S]


class TerminalARDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=4, n_heads=4):
        super().__init__()
        self.input_emb = nn.Linear(input_dim, hidden_dim)
        self.context_emb = nn.Linear(input_dim, hidden_dim)
        self.contact_emb = nn.Embedding(4, hidden_dim) 
        
        # Learnable Positional Embedding
        self.pos_emb = nn.Embedding(500, hidden_dim) 

        # Standard PyTorch Transformer Decoder (implements "Attention Is All You Need")
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.mog_head = MixtureOfGaussiansHead(hidden_dim, input_dim)

    def forward(self, x0_curr, x1, c, y_emb):
        """
        x0_curr: Shifted clean motion (autoregressive input)
        x1: Noisy motion (context)
        c: Contact latents (context)
        y_emb: Text embedding (context)
        """
        B, S, D = x0_curr.shape
        
        # 1. Embed Input (Motion + Contact + Pos)
        x_emb = self.input_emb(x0_curr)
        # Sum contact embeddings for Left and Right feet
        c_emb = self.contact_emb(c[..., 0]) + self.contact_emb(c[..., 1])
        # Positional encoding
        positions = torch.arange(S, device=x0_curr.device).unsqueeze(0)
        
        tgt = x_emb + c_emb + self.pos_emb(positions)

        # 2. Build Context Memory (Text + Noisy x1)
        x1_emb = self.context_emb(x1)
        # y_emb is [B, 1, D]. Expand to match S if needed, or let attention handle it.
        # We concatenate them: Memory = [Text_Token, x1_Frame_0, x1_Frame_1, ...]
        memory = torch.cat([y_emb.expand(-1, S, -1), x1_emb], dim=1) 

        # 3. Causal Mask (Ensures frame t only sees 0...t)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x0_curr.device)
        
        # 4. Decode
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.mog_head(out)
    
    def get_landmarks(self, vertices):
        """ vertices: [B, Frames, 6890, 3] """
        landmarks = {}
        for name, indices in self.vertex_indices.items():
            idx = torch.tensor(indices, device=vertices.device, dtype=torch.long)
            # Average the vertices to find the centroid of the contact patch
            landmarks[name] = vertices[:, :, idx, :].mean(dim=2) 
        return landmarks

    def compute_heights(self, landmarks):
        heights = {}
        for name, pos in landmarks.items():
            # Dot product: (p - p0) . n
            h = ((pos - self.ground_point) * self.ground_normal).sum(dim=-1)
            heights[name] = h
        return heights

    def compute_tangential_velocities(self, landmarks, dt=1/30.0):
        velocities = {}
        for name, pos in landmarks.items():
            # v = (p_{t+1} - p_t) / dt
            delta = (pos[:, 1:] - pos[:, :-1]) / dt
            # Project onto ground plane: v_par = v . P
            v_par = torch.matmul(delta, self.P_parallel.t())
            # Return squared magnitude ||v||^2
            velocities[name] = (v_par ** 2).sum(dim=-1)
        return velocities


# --- 2. Contact Networks (Prior and Posterior) ---
class ContactNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_on_text=False):
        super().__init__()
        self.condition_on_text = condition_on_text
        effective_input = input_dim + hidden_dim if condition_on_text else input_dim
        
        self.net = nn.Sequential(
            nn.Linear(effective_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 4 * 2) # 4 logits (None, Heel, Toe, Both) * 2 Feet
        )

    def forward(self, x_motion, y_emb=None):
        # x_motion: [B, S, D]
        if self.condition_on_text:
            # y_emb: [B, 1, D] -> Expand to [B, S, D]
            y_repeated = y_emb.expand(-1, x_motion.shape[1], -1)
            x_in = torch.cat([x_motion, y_repeated], dim=-1)
        else:
            x_in = x_motion
        return self.net(x_in).view(x_motion.shape[0], x_motion.shape[1], 2, 4)


# --- 3. Terminal AR Decoder components ---
class MixtureOfGaussiansHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_mixtures=10):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.fc_pi = nn.Linear(hidden_dim, num_mixtures)
        self.fc_mu = nn.Linear(hidden_dim, num_mixtures * output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_mixtures * output_dim)

    def forward(self, x):
        B, S, _ = x.shape
        pi_logits = self.fc_pi(x)
        mu = self.fc_mu(x).view(B, S, self.num_mixtures, self.output_dim)
        sigma = torch.exp(torch.clamp(self.fc_sigma(x), min=-10, max=2)).view(B, S, self.num_mixtures, self.output_dim)
        return pi_logits, mu, sigma

    def log_prob(self, x_target, pi_logits, mu, sigma):
        # Calculates Log-Likelihood for loss
        x_target = x_target.unsqueeze(2)
        var = sigma ** 2
        log_prob = -0.5 * (torch.log(2 * np.pi * var) + (x_target - mu)**2 / var)
        log_prob = torch.sum(log_prob, dim=-1) # Sum over feature dims
        # LogSumExp over mixtures
        return torch.logsumexp(torch.log_softmax(pi_logits, dim=-1) + log_prob, dim=-1)


class TerminalARDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=4, n_heads=4):
        super().__init__()
        self.input_emb = nn.Linear(input_dim, hidden_dim)
        self.context_emb = nn.Linear(input_dim, hidden_dim)
        self.contact_emb = nn.Embedding(4, hidden_dim) 
        
        # Positional Encoding
        self.pos_emb = nn.Embedding(500, hidden_dim) 

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.mog_head = MixtureOfGaussiansHead(hidden_dim, input_dim)

    def forward(self, x0_curr, x1, c, y_emb):
        B, S, D = x0_curr.shape
        
        # 1. Embed Input (Motion + Contact + Pos)
        x_emb = self.input_emb(x0_curr)
        c_emb = self.contact_emb(c[..., 0]) + self.contact_emb(c[..., 1])
        positions = torch.arange(S, device=x0_curr.device).unsqueeze(0)
        
        # Shape: [Batch, S, Hidden]
        tgt = x_emb + c_emb + self.pos_emb(positions)

        # 2. Build Context Memory (Text + Noisy x1)
        x1_emb = self.context_emb(x1)
        # Memory Shape: [Batch, S+1, Hidden]
        memory = torch.cat([y_emb.expand(-1, S, -1), x1_emb], dim=1) 

        # 3. Causal Mask
        tgt_mask = torch.zeros(S, S, device=x0_curr.device)
        future_mask = torch.triu(torch.ones(S, S, device=x0_curr.device), diagonal=1).bool()
        tgt_mask = tgt_mask.masked_fill(future_mask, float('-inf'))     

        if torch.rand(1).item() < 0.001: # Print very rarely
            print(f"\n[DIM CHECK] AR Decoder Inputs")
            print(f"  > Input (Batch, Time, Dim): {x0_curr.shape}")
            print(f"  > Target Mask: {tgt_mask.shape}")
            # Verify we are about to transpose CORRECTLY for PyTorch 1.7
            print(f"  > Transposing (0,1)... Expecting [Time, Batch, Dim]")
   
        # Old PyTorch expects the first dimension to be the Sequence/Time length
        tgt = tgt.transpose(0, 1)       # [S, B, H]
        memory = memory.transpose(0, 1) # [S+1, B, H]

        # 4. Decode
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        
        out = out.transpose(0, 1)       # [B, S, H]

        return self.mog_head(out)