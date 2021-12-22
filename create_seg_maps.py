"""
Visualize the projections in published HO-3D dataset

python create_seg_maps.py /home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3 /home/rafay_veeve/Desktop/Veeve/galactus/YCB_Video_Models -split train -visType open3d

"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
import os
from pathlib import Path
from copy import deepcopy
import open3d

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
    install('chumpy')
    import chumpy as ch


try:
    import pickle
except:
    install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


if __name__ == '__main__':

    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
    ap.add_argument("-split", required=False, type=str,
                    help="split type", choices=['train', 'evaluation'], default='train')
    ap.add_argument("-seq", required=False, type=str,
                    help="sequence name")
    ap.add_argument("-id", required=False, type=str,
                    help="image ID")
    ap.add_argument("-visType", required=False,
                    help="Type of visualization", choices=['open3d', 'matplotlib'], default='open3d')
    ap.add_argument("-cmap", required=False, help="The color map of the palette to apply to generated segmentation masks")
    args = vars(ap.parse_args())

    #FIXME: get this from args
    palette = np.load("/home/veeve/workspace/Rafay/galactus/cmap.npy")

    baseDir = Path(args['ho3d_path'])
    YCBModelsDir = Path(args['ycbModels_path'])
    split = args['split']

    print(os.listdir(baseDir/split))
    # store base path of the datset train or split
    data_base_path = baseDir/split

    # iterate over all the sequences in the dataset
    for seqName in os.listdir(data_base_path):

        print(os.listdir(data_base_path/seqName))
        print(seqName)

        # store full sequence path
        full_seq_path = data_base_path/seqName

        # create seg directory
        if not os.path.exists(data_base_path/seqName/"segmentations"):
            os.mkdir(data_base_path/seqName/"segmentations")

        # go over all the images
        for img_name in sorted(os.listdir(data_base_path/seqName/"rgb")):

            # extract id from image_name
            id = img_name.split(".")[0]
            
            file_to_check = Path(str(data_base_path/seqName/"segmentations"/id) + ".png")
            if file_to_check.is_file():
                print(f"{id}.png already exists")
                continue
            
            print(id)
            
            # read image, depth maps and annotations
            img = read_RGB_img(baseDir, seqName, id, split)
            depth = read_depth_img(baseDir, seqName, id, split)
            anno = read_annotation(baseDir, seqName, id, split)

            if anno['objRot'] is None:
                print('Frame %s in sequence %s does not have annotations'%(args['id'], args['seq']))
                continue

            # get object 3D corner locations for the current pose
            objCorners = anno['objCorners3DRest']
            objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
            # get the hand Mesh from MANO model for the current pose
            if split == 'train':
                handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
            # project to 2D
            if split == 'train':
                handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
            else:
                # Only root joint available in evaluation split
                handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)
            objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)

            # Visualize
            if args['visType'] == 'open3d':
                # open3d visualization
                if not os.path.exists(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')):
                    raise Exception('3D object models not available in %s'%(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')))
                # load object model
                objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))
                # apply current pose to the object model
                objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
                # show
                if split == 'train':
                    # open3dVisualize([handMesh, objMesh], ['r', 'g'])
                    saveOpen3dVisualization([handMesh, objMesh], ['r', 'g'], anno['camMat'],
                                            palette,
                                            save_loaction=data_base_path / seqName / "segmentations" / id)
                else:
                    saveOpen3dVisualization([objMesh], ['r', 'g'], anno['camMat'],
                                            palette,
                                            save_loaction=data_base_path / seqName / "segmentations" / id)
