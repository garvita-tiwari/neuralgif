# -*- coding: UTF-8 -*-
"""source: https://github.com/chaitanya100100/TailorNet"""
import numpy as np
import os
import copy
import mathutils



BLEND_FILE = '/BS/garvita2/work/vis_files/scene.blend'
def blender_render(meshes_path, tex_num, outpath, min_val, smooth):
    import bpy

    scene = bpy.context.scene

    bpy.context.scene.render.resolution_x = 2048
    bpy.context.scene.render.resolution_y = 2048

    scene.render.resolution_percentage = 100
    scene.render.use_border = False

    scene.render.alpha_mode = 'TRANSPARENT'
    names = []
    for idx in range(len(meshes_path)):
        imported_object = bpy.ops.import_scene.obj(filepath=meshes_path[idx], axis_forward='-Z', axis_up='Y')
        obj_object = bpy.context.selected_objects[0]  ####<--Fix
        names.append(obj_object.name)
        # smooth shading
        for f in obj_object.data.polygons:
            f.use_smooth = True
        for edge in obj_object.data.edges:
            edge.use_edge_sharp = False
        print(tex_num[idx])
        print("NO SHADOWS")
        #if tex_num[idx] == 0:  # body, object of interest
        if idx == 0:  # body, object of interest
            mat = bpy.data.materials['Material.006']
            #obj_object.data.materials.append(mat)
            mat = mat.copy()
            mat.diffuse_color = (1.0, 1.0, 1.0)
            mat.specular_color = (0.0, 0.0, 0.0)
            #mat.diffuse_color = (0.6, 1, 0.8)
            # mat.use_cast_shadows = False
        elif idx == 1:  # tshirt , new body
            # mat = mat.copy()
            # mat.diffuse_color = (0.6, 1, 0.8)
            # mat.specular_color = (0.0, 0.0, 0.0)
            mat = bpy.data.materials['Material.007']
            obj_object.data.materials.append(mat)
            #mat = bpy.data.materials['Material.006']
            # mat.use_cast_shadows = False
        elif tex_num[idx] == 2:  # shirt
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (0.6, 1, 0.8)
        elif tex_num[idx] == 3:  # Pants
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (1.0, 0.4, 0.4)
        elif tex_num[idx] == 4:  # skirt
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (0.35, 0.4, 1.0)
        else:
            raise AttributeError

        if smooth =='True':
            bpy.ops.object.shade_smooth()
    bpy.data.objects['Grid'].select = True
    obj = bpy.data.objects['Grid'] #
    (x, y, z) = (0.0, 0.0, min_val)

    # adding adjustment values to the property
    obj.location.z =  min_val
    print(obj.name)
    # to select the object in the 3D viewport,
    # this way you can also select multiple objects

    # additionally you can use
    #bpy.context.scene.objects.active = bpy.data.objects['Sphere.017']
    bpy.context.scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)
    for nm in names:
        objs = bpy.data.objects
        objs.remove(objs[nm], do_unlink=True)


def get_rotmat(side):
    from scipy.spatial.transform import Rotation as R
    if side == 'front':
        s = R.from_rotvec((0.) * np.array([1, 0, 0]))
    elif side == 'back':
        s = R.from_rotvec((np.pi) * np.array([0, 1, 0]))
    elif side.startswith("right"):
        angle = side.replace("right", "")
        s = R.from_rotvec((float(angle) * np.pi / 180) * np.array([0, 1, 0]))
    return s.as_dcm()


def preproc_garbody(gar, body, garment_class=None, side='front'):
    gar = copy.copy(gar)
    body = copy.copy(body)

    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)
    body.v = body.v.dot(rotmat)

    miny = body.v[:, 1].min() + 0.72

    gar.v[:, 1] -= miny
    body.v[:, 1] -= miny
    return gar, body


def visualize_garment_body(gar, body, outpath, garment_class='t-shirt', side='front', out_folder='.', smooth=False):
    # gar = copy.copy(gar)
    # body = copy.copy(body)
    #
    # rotmat = get_rotmat(side)
    # gar.v = gar.v.dot(rotmat)
    # body.v = body.v.dot(rotmat)
    #
    # miny = body.v[:, 1].min() + 0.72
    #
    # gar.v[:, 1] -= miny
    # body.v[:, 1] -= miny
    #
    # gar_path = "/tmp/gar.obj"
    # body_path = "/tmp/body.obj"
    #
    thispath = os.path.abspath(__file__)
    thispath = '/BS/garvita/work/code/can_pose/visualisation/blender_renderer.py'
    body = copy.copy(body)
    gar = copy.copy(gar)
    rotmat = get_rotmat(side)
    body.v = body.v.dot(rotmat)
    gar.v = gar.v.dot(rotmat)

    z_min = np.min(body.v[:,1])
    body_path = out_folder + "/body.obj"
    gar_path = out_folder + "/gar.obj"


    if gar_path.endswith("obj"):
        gar.write_obj(gar_path)
        body.write_obj(body_path)
    else:
        gar.write_ply(gar_path)
        body.write_ply(body_path)

    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {} -- --body {} --gar {}  --min_val {} --outpath {} --gar_classes {} --smooth {}".format(
        BLEND_FILE,
        thispath,
        gar_path,
        body_path,
        z_min,
        outpath,
        garment_class,
        smooth
    )
    print(cmd)
    os.system(cmd)
    print("Done")

def visualize_joints(gar, body, outpath, garment_class='t-shirt', side='front', out_folder='.', smooth=False):

    thispath = os.path.abspath(__file__)
    thispath = '/BS/garvita/work/code/can_pose/visualisation/blender_renderer.py'
    gar = copy.copy(gar)
    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)
    gar_path = out_folder + "/gar.obj"
    gar.write_obj(gar_path)
    all_jts_path = []
    for i, jt in enumerate(body):
        jt = copy.copy(jt)
        jt.v = jt.v.dot(rotmat)
        body_path = out_folder + "/{}.obj".format(i)
        all_jts_path[body_path]
        jt.write_obj(body_path)

    z_min = np.min(body.v[:,1])


    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {} -- --body {} --gar {} --min_val {} --outpath {} --gar_classes {} --smooth {}".format(
        BLEND_FILE,
        thispath,
        gar_path,
        body_path,
        z_min,
        outpath,
        garment_class,
        smooth
    )
    print(cmd)
    os.system(cmd)
    print("Done")


def visualize_garment(gar, outpath, garment_class='t-shirt', side='front'):
    import os
    import copy
    gar = copy.copy(gar)

    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)

    gar_path = "/tmp/gar.obj"
    if gar_path.endswith("obj"):
        gar.write_obj(gar_path)
    else:
        gar.write_ply(gar_path)

    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {} -- --gar {} --outpath {} --gar_classes {}".format(
        BLEND_FILE,
        thispath,
        gar_path,
        outpath,
        garment_class,
    )
    os.system(cmd)
    print("Done")


def visualize_body(body, outpath, side='front', out_folder='.', smooth=False):
    import os
    import copy
    thispath = os.path.abspath(__file__)
    #thispath = '/BS/garvita/work/code/can_pose/visualisation/blender_renderer.py'
    body = copy.copy(body)
    # import ipdb
    # ipdb.set_trace()
    rotmat = get_rotmat(side)
    body.v = body.v.dot(rotmat)
    z_min = np.min(body.v[:,1])
    body_path = out_folder + "/body.obj"
    if body_path.endswith("obj"):
        body.write_obj(body_path)
    else:
        body.write_ply(body_path)
    cmd = "blender --background {} -P {} -- --body {} --min_val {} --outpath {} --smooth {}".format(
        BLEND_FILE,
        thispath,
        body_path,
        z_min,
        outpath,
        side,
        smooth
    )
    print(cmd)
    os.system(cmd)
    print("Done")


if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0]] + sys.argv[6:]
    print(sys.argv)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath')
    parser.add_argument('--smooth')
    parser.add_argument('--min_val', default=0.0, type=float)
    parser.add_argument('--body', nargs='+')
    parser.add_argument('--gar', nargs='+')
    parser.add_argument('--gar_classes', nargs='+')
    args = parser.parse_args()
    print(args)

    # assert(len(args.body) == len(args.gar))
    tex_num = []
    meshes = []

    dd = {
        't-shirt': 1,
        'old-t-shirt': 1,
        'shirt': 2,
        'pants': 3,
        'skirt': 4,
    }

    if args.body is not None:
        tex_num += [0] * len(args.body)
        meshes += args.body
    if args.gar is not None:
        meshes += args.gar
        if args.gar_classes is None:
            tex_num += [1] * len(args.gar)
        else:
            tex_num += [dd[gc] for gc in args.gar_classes]
    blender_render(meshes, tex_num=tex_num, outpath=args.outpath, min_val=args.min_val, smooth=args.smooth)