import torch
import torch.nn as nn
import torch.nn.functional as F

class SMPLGeometryWrapper(nn.Module):
    """
    Handles geometric calculations for SMPL-based body representations.
    Utilizes fixed offsets for Virtual Markers (Toes/Heels) relative to foot joints.
    """
    def __init__(self):
        super().__init__()
        
        # Vertex indices (Kept for reference or mesh-based operations)
        self.vertex_indices = {
            'L_Toe': [3218, 3219, 3220, 3227, 3238, 3240, 3250, 3253, 3262, 3263, 3275, 3278, 3293, 3295, 3305, 3306],
            'L_Heel': [3458, 3459, 3460, 3461, 3463, 3466, 3467, 3468],
            'R_Toe': [6615, 6620, 6622, 6630, 6635, 6639, 6650, 6654, 6659, 6664, 6674, 6677, 6693, 6695, 6705, 6706],
            'R_Heel': [6858, 6859, 6860, 6861, 6864, 6866, 6867, 6868]
        }
        
        # STATIC OFFSETS: Relative vector from Ankle Joint to Virtual Marker
        # Coords: [X, Y, Z]
        self.register_buffer('offset_l_toe',  torch.tensor([-0.00957836, -0.01430488,  0.05481965]))
        self.register_buffer('offset_l_heel', torch.tensor([-0.01765292, -0.00870803, -0.15520199]))
        self.register_buffer('offset_r_toe',  torch.tensor([ 0.0088405,  -0.01663555,  0.05654152]))
        self.register_buffer('offset_r_heel', torch.tensor([ 0.01887021, -0.00870501, -0.15856517]))

        # Ground plane definitions
        self.register_buffer('ground_normal', torch.tensor([0., 1., 0.]))
        self.register_buffer('ground_point', torch.tensor([0., 0., 0.]))
        # Projection matrix for tangential velocity (removing Y component)
        self.register_buffer('P_parallel', torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]))

    def get_landmarks_from_skeleton(self, skeleton_xyz):
        """
        Calculates virtual landmarks (Toes/Heels) by applying fixed offsets to ankle joints.
        skeleton_xyz: [Batch, Joints, 3, Frames] (Standard HumanML3D format)
        Returns: Dict of [Batch, Frames, 3]
        """
        # HML3D Standard Indices: 7=L_Ankle, 8=R_Ankle
        # Extract and permute to [B, T, 3]
        l_ankle = skeleton_xyz[:, 7, :, :].permute(0, 2, 1)
        r_ankle = skeleton_xyz[:, 8, :, :].permute(0, 2, 1)

        landmarks = {
            'L_Toe':  l_ankle + self.offset_l_toe,
            'L_Heel': l_ankle + self.offset_l_heel,
            'R_Toe':  r_ankle + self.offset_r_toe,
            'R_Heel': r_ankle + self.offset_r_heel
        }
        return landmarks

    def compute_heights(self, landmarks):
        """ Returns signed height (distance to ground) for each landmark. """
        heights = {}
        for name, pos in landmarks.items():
            # Dot product: (p - p0) . n
            # pos: [B, T, 3], ground_normal: [3] -> [B, T]
            h = ((pos - self.ground_point) * self.ground_normal).sum(dim=-1)
            heights[name] = h
        return heights

    def compute_tangential_velocities(self, landmarks, dt=1/20.0):
        """ Returns squared magnitude of tangential velocity on the ground plane. """
        velocities = {}
        for name, pos in landmarks.items():
            # Finite difference: v = (p_{t+1} - p_t) / dt
            # Note: This reduces sequence length by 1. We usually pad or slice masks accordingly.
            delta = (pos[:, 1:] - pos[:, :-1]) / dt
            
            # Project onto ground plane: v_par = v . P
            # [B, T-1, 3] @ [3, 3] -> [B, T-1, 3]
            v_par = torch.matmul(delta, self.P_parallel.t())
            
            # Return squared magnitude ||v||^2
            velocities[name] = (v_par ** 2).sum(dim=-1)
        return velocities
