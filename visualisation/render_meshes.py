import os
import ipdb
from psbody.mesh import Mesh
from blender_renderer import visualize_body
import numpy as np


def render_meshes(mesh_path, out_dir, mesh_name=None):
    if mesh_name is None:
        #render whole directory
        for i in range( 893):
            # if os.path.exists(os.path.join(out_dir, "{:06d}_front.png".format(i))):
            #     continue
            frame_name = 'walk_{:06}'.format(i)
            frame_name = 'walk_{:03}'.format(i)
            frame_name = '{:03}'.format(i)
            org_mesh = os.path.join(mesh_path, '{}.obj'.format(frame_name))
            #org_mesh = os.path.join(mesh_path, 'walk_{}.obj'.format(frame_name))
            if not os.path.exists(org_mesh):
                continue
            body = Mesh(filename=org_mesh)
            body.v += np.array([0, 0.5, 0])
            visualize_body(
                body, os.path.join(out_dir, "{:06d}_front.png".format(i)), side='front', out_folder=out_dir)

def render_meshes_byname(mesh_path, out_dir, mesh_name=None):
    if mesh_name is None:
        #render whole directory
        all_names = sorted(os.listdir(mesh_path))
        for frame_name  in all_names:
            frame_name = frame_name[:-4]

            if os.path.exists(os.path.join(out_dir, "{}_front.png".format(frame_name))):
                continue
            if '.' in frame_name:
                continue
            print(frame_name)
            org_mesh = os.path.join(mesh_path, '{}.obj'.format(frame_name))
            #org_mesh = os.path.join(mesh_path, 'walk_{}.obj'.format(frame_name))
            if not os.path.exists(org_mesh):
                continue
            body = Mesh(filename=org_mesh)
            body.v += np.array([0, 0.5, 0])
            visualize_body(
                body, os.path.join(out_dir, "{}_front.png".format(frame_name)), side='front', out_folder=out_dir)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='render results using blender'
    )
    parser.add_argument('-mesh_path', default='/BS/RVH_3dscan_raw2/static00/ipdf/data/walk_trail1/rand_meshes_0.3', type=str, help='Path to mesh directory.')
    parser.add_argument('-out_dir',  default='/BS/RVH_3dscan_raw2/static00/ipdf/data/walk_trail1/rand_meshes_0.3_vis', type=str, help='Path to output directory.')
    parser.add_argument('-mesh_name',  type=str, help='filename without extension.')

    args = parser.parse_args()

    render_meshes(args.mesh_path, args.out_dir, args.mesh_name)
