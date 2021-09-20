import os
import trimesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import Pool


# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)


    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    data_tupels = []
    for i, data in tqdm(enumerate(loader)):

        path = os.path.normpath(data['path'][0])
        # print(out_path, path)

        if os.path.exists(out_path + data['path'][0] +'.obj'):
            print('Path exists - skip! {}'.format(out_path + data['path'][0] +'.obj'))
            continue

        try:
            if len(data_tupels) > 20:
                create_meshes(data_tupels)
                data_tupels = []
            logits, min, max,can_pt = gen.generate_mesh(data)
            data_tupels.append((logits,min, max, can_pt, data, out_path))


        except Exception as err:
            print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))

    try:

        create_meshes(data_tupels)
        data_tupels = []
        logits,  min, max,can_pt = gen.generate_mesh(data)
        data_tupels.append((logits,  min, max,can_pt, data, out_path))


    except Exception as err:
        print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))


def save_mesh(data_tupel):
    logits,  min, max , can_pt, data, out_path = data_tupel

    if not os.path.exists(out_path + data['path'][0] +'.obj'):

        mesh, can_mesh = gen.mesh_from_logits(logits, min, max , can_pt)

        # path = os.path.normpath(data['path'][0])
        # export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

        path = os.path.normpath(data['path'][0])
        #export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])
        export_path = out_path + '/{}/'.format(path.split(os.sep)[-2] )
        print('saving mesh')
        if len(path.split(os.sep)[-2]) == 0:
            export_path = out_path
        if not os.path.exists(export_path):
            try:
                os.makedirs(export_path)
            except:
                print('folder already exists')

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        try:
            mesh.export(out_path + data['path'][0] +'.obj')
            can_mesh.export(out_path + data['path'][0] +'_can.obj')
        except:
            mesh.export(out_path + '/' + data['path'][0] +'.obj')
            can_mesh.export(out_path  + '/' +  data['path'][0] +'_can.obj')
    else:
        can_mesh = trimesh.Trimesh(can_pt)
        can_mesh.export(out_path + data['path'][0] + '_can.obj')
def create_meshes(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()