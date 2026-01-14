import torch
import torch.nn as nn

class SMPLGeometryWrapper(nn.Module):
    """
    Minimalist Geometry Wrapper.
    Uses only SMPL Joints 10 (Left Foot) and 11 (Right Foot) to detect contact/sliding.
    """
    def __init__(self):
        super().__init__()
        
        # Ground plane definitions (assuming y=0 flat floor)
        self.register_buffer('ground_normal', torch.tensor([0., 1., 0.]))
        self.register_buffer('ground_point', torch.tensor([0., 0., 0.]))
        
        # Projection matrix for tangential velocity (removes Y component)
        self.register_buffer('P_parallel', torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]))

    def get_landmarks_from_skeleton(self, skeleton_xyz):
        """
        Extracts only the Foot joints (10/11).
        skeleton_xyz: [Batch, Joints, 3, Frames]
        Returns: Dict of [Batch, Frames, 3]
        """
        # HML3D/SMPL Topology:
        # 10 = Left Foot
        # 11 = Right Foot
        
        # Permute to [Batch, Frames, 3]
        l_foot = skeleton_xyz[:, 10, :, :].permute(0, 2, 1)
        r_foot = skeleton_xyz[:, 11, :, :].permute(0, 2, 1)

        landmarks = {
            'L_Foot': l_foot,
            'R_Foot': r_foot
        }
        return landmarks

    def compute_heights(self, landmarks):
        """ Returns signed height (distance to ground) for each foot joint. """
        heights = {}
        for name, pos in landmarks.items():
            # Dot product: (p - p0) . n
            h = ((pos - self.ground_point) * self.ground_normal).sum(dim=-1)
            heights[name] = h
        return heights

    def compute_tangential_velocities(self, landmarks, dt=1/20.0):
        """ Returns squared magnitude of tangential velocity on the ground. """
        velocities = {}
        for name, pos in landmarks.items():
            # Finite difference velocity
            delta = (pos[:, 1:] - pos[:, :-1]) / dt
            
            # Project onto ground plane (ignore vertical motion)
            v_par = torch.matmul(delta, self.P_parallel.t())
            
            # Squared magnitude
            velocities[name] = (v_par ** 2).sum(dim=-1)
        return velocities
