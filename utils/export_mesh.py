import torch
import trimesh
import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.rotation2xyz import Rotation2xyz 

def export_reference_mesh():
    print("Initializing SMPL model...")
    # Initialize loader to access the internal SMPL layer
    rot2xyz = Rotation2xyz(device='cpu', dataset='amass') 
    smpl_layer = rot2xyz.smpl_model

    print("Creating T-pose (Identity Matrices)...")
    
    # 1. Create Identity Matrix (3x3)
    # Shape: [1, 3, 3]
    identity_mat = torch.eye(3).unsqueeze(0)

    # 2. Global Orientation: 
    # Needs shape [Batch=1, 1, 3, 3]
    global_orient = identity_mat.unsqueeze(1) 
    
    # 3. Body Pose: 
    # Needs shape [Batch=1, 23, 3, 3]
    # We repeat the identity matrix 23 times
    body_pose = identity_mat.unsqueeze(1).repeat(1, 23, 1, 1)
    
    # 4. Betas (Shape)
    betas = torch.zeros(1, 10)

    # 5. Forward pass directly on the SMPL layer
    # We explicitly set pose2rot=False to tell SMPL "These are already matrices"
    output = smpl_layer(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        return_verts=True,
        pose2rot=False 
    )
    
    vertices = output['vertices'] # Shape: [1, 6890, 3]
    
    # 6. Save to OBJ
    print("Exporting to OBJ...")
    faces = smpl_layer.faces
    mesh = trimesh.Trimesh(vertices=vertices[0].detach().numpy(), faces=faces)
    
    output_path = 'smpl_reference.obj'
    mesh.export(output_path)
    print(f"Success! Exported T-pose to {output_path}")

if __name__ == "__main__":
    export_reference_mesh()