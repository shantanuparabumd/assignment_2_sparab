

import code.losses as losses
from pytorch3d.utils import ico_sphere
from code.r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import code.dataset_location as dataset_location
import torch
from code.utils import get_device, get_mesh_renderer, get_points_renderer
from PIL import Image, ImageDraw
import numpy as np
import imageio
import pytorch3d
from tqdm.auto import tqdm

def generate_gif_from_voxels(voxels,path):

    frames = 72
    my_images = []

    camera_poses = get_camera_poses(distance=3.0,elevation=0.0,number_of_views=frames)
    for pose in tqdm(camera_poses):
        rend = render_voxels(voxels,camera_rotation=pose[0],camera_translation=pose[1])
        image = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(image))

    imageio.mimsave(path, my_images, duration=0.2)



def render_voxels(voxels,image_size=256, color=[0.1, 0.7, 1], device=None, camera_rotation = None, camera_translation = None):

    mesh = pytorch3d.ops.cubify(voxels,thresh = 0.5)


    if device is None:
        device = get_device()


    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = torch.tensor(mesh.verts_list()[0]),torch.tensor(mesh.faces_list()[0])
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices).to(device)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)


    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_rotation, T=camera_translation, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend



def get_camera_poses(distance=3.0,elevation=0.0,number_of_views=36):

    # List to store all the camera poses
    camera_poses=[]
    device = get_device()
   

    for i in range(number_of_views):
        # Compute Azimuth
        # Angle created by the projection of the vector from camera line of sight to object with reference vector
        azimuth = (i/number_of_views)*360.0

        # Compute the view transform for the current view
        view_transform = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=distance,
            elev=elevation,
            azim=azimuth,
            degrees="True",
            device=device
        )
        
        
        # Append the view transform to the list
        camera_poses.append(view_transform)

    return camera_poses

def generate_gif_from_point(
    point_cloud,path,
    image_size=256,
    background_color=(1, 1, 1),
    device=None
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud=point_cloud.squeeze(0)
    
    rgba = torch.tensor([[0.588, 0.243, 0.419, 1.0]]).to(device)

    verts = torch.Tensor(point_cloud).unsqueeze(0).to(device)
    rgba = rgba.repeat(1,verts.shape[1], 1)
    print(rgba.shape)
    
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgba).to(device)
    
    # Place a point light in front of the plant.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)
    frames = 144
    my_images = []

    
    camera_poses = get_camera_poses(distance=3.0,number_of_views=frames)
    for pose in tqdm(camera_poses):
        cameras_360 = pytorch3d.renderer.FoVPerspectiveCameras(
            R=pose[0], T=pose[1], fov=60, device=device
        )
        rend = renderer(point_cloud, cameras=cameras_360, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)  # (B, H, W, 4) -> (H, W, 3)
        image = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(image))

    

    imageio.mimsave(path, my_images, duration=10.0)


def render_mesh(
    mesh_src, image_size=256, color=[0.7, 0.7, 1], device=None, camera_rotation = None, camera_translation = None
):

    if device is None:
        device = get_device()


    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = torch.tensor(mesh_src.verts_list()[0]),torch.tensor(mesh_src.faces_list()[0])
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices).to(device)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)


    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_rotation, T=camera_translation, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def generate_gif_from_mesh(mesh_src,path):

    frames = 72
    my_images = []

    camera_poses = get_camera_poses(distance=3.0,elevation=0.0,number_of_views=frames)
    for pose in tqdm(camera_poses):
        rend = render_mesh(mesh_src,camera_rotation=pose[0],camera_translation=pose[1])
        image = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(image))

    imageio.mimsave(path, my_images, duration=0.2)