# -*- coding: utf-8 -*-
# This is to sample rotations for the inference template poses \bar(R), or the training reference rotations R_c used in geometric prior
# For each sampled rotation, its rendered BGR image, edgemap, 2D boundingbox, and rotations will be generated and preserved
# これは推論テンプレートのポーズ\bar(R)、または幾何学的事前処理で使用される学習参照回転R_cの回転をサンプリングするものです。
# サンプリングされた各回転に対して、そのレンダリングされたBGR画像、エッジマップ、2Dバウンディングボックス、回転が生成・保存されます。
import numpy as np
import progressbar
import cv2
import os,sys

from pysixd_stuff.pysixd import inout
from pysixd_stuff.pysixd import renderer_vt
from pysixd_stuff.pysixd import view_sampler
import data_utils

def generate_codebook_imgs(path_model,dir_imgs,dir_edges, path_obj_bbs,path_rot,render_dims,cam_K,depth_scale=1.,texture_img=None,start_end=None):
    if not os.path.exists(dir_imgs): # RGB画像の保存先にdirがなければ作成する．
        os.makedirs(dir_imgs)
    if not os.path.exists(dir_edges): # Edge画像の保存先にdirがなければ作成する．
        os.makedirs(dir_edges)

    view_Rs=data_utils.viewsphere_for_embedding_v2(num_sample_views=2500,num_cyclo=36,use_hinter=True)
    #data_utils.viewsphere_for_embedding_v2(num_sample_views=2500,num_cyclo=36,use_hinter=True)
    #For reference R_c: view_Rs=data_utils.viewsphere_for_embedding_v2(num_sample_views=400,num_cyclo=20,use_hinter=False)

    #num_sample_views: number of samples on the unit sphere
    #num_cyclo: number of samples regarding inner-plane rotations
    #use_hinter=True: hinter sampling; use_hinter=False: fabonicci sampling


    np.savez(path_rot,rots=view_Rs)
    embedding_size = view_Rs.shape[0]

    out_shape=(128,128,3)
    if start_end and start_end[0]!=0:
        obj_bbs=np.load(path_obj_bbs+'.npy')
    else:
        obj_bbs = np.empty((embedding_size, 4))
    print('Creating embedding ..')
    bar = progressbar.ProgressBar(
        maxval=embedding_size,
        widgets=[' [', progressbar.Timer(), ' | ', progressbar.Counter('%0{}d / {}'.format(len(str(embedding_size)), embedding_size)), ' ] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ']
    )
    bar.start()
    K = np.array(cam_K).reshape(3,3)

    clip_near = float(10)
    clip_far = float(10000)
    pad_factor = float(1.2)

    t = np.array([0, 0, float(700)])
    model = inout.load_ply(path_model) # PLYファイルから3Dメッシュモデルを読み込む
    model['pts']*=depth_scale 

    if start_end is None:
        search_range=range(0,view_Rs.shape[0])
    else:
        start_end[1]=min(start_end[1],view_Rs.shape[0])
        search_range=range(start_end[0],start_end[1])

    for i in search_range:
        bar.update(i)
        R=view_Rs[i]
        rgb_y, depth_y = renderer_vt.render_phong(model, render_dims, K.copy(), R, t, clip_near=clip_near,
                                              clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=False)
        ys, xs = np.nonzero(depth_y > 0)
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

        obj_bbs[i] = obj_bb
        bgr_y = rgb_y.copy()
        for cc in range(0, 3):
            bgr_y[:, : ,cc] = rgb_y[:, :,2 - cc]

        resized_bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor,resize=out_shape[:2],interpolation = cv2.INTER_NEAREST)
        resized_bgr_y_edge=cv2.Canny(resized_bgr_y,50,150)
        cv2.imwrite(os.path.join(dir_edges,'{:05d}.png'.format(i)),resized_bgr_y_edge)
        cv2.imwrite(os.path.join(dir_imgs,'{:05d}.png'.format(i)),resized_bgr_y)
    bar.finish()
    np.save(path_obj_bbs,obj_bbs)


if __name__=='__main__':
	#Rendered by batch, with batch size=50
    obj_id=int(sys.argv[1]) # 物体id:1-30
    bid=int(sys.argv[2]) # batch_id:謎
    batch_size=50
    path_model ='./ws/meshes/obj_{:02d}.ply'.format(obj_id) # Path of the 3D mesh ply file． 3dモデルのplyファイルの指定
    dir_out='./embedding92232s/{:02d}/'.format(obj_id) # レンダリング画像（RGB・Edge）や物体検出時の2dBBoxの情報，サンプル？の回転情報を保存するdirのパス
    dir_imgs = os.path.join(dir_out,'imgs') # ./embedding92232s/00/imgs
    dir_edges= os.path.join(dir_out,'in_edges2') # ./embedding92232s/00/in_edges2
    path_obj_bbs= os.path.join(dir_out,'obj_bbs') # ./embedding92232s/00/obj_bbs
    path_rot=os.path.join(dir_out,'rot_infos') # ./embedding92232s/00/rot_infos

    path_texture=None # T-LESSなのでTextureなし
    texture_img_rgb=None # T-LESSなのでTextureのRGB画像もなし
    if path_texture: # 今回はNoneなので無視
        texture_img_bgr=cv2.imread(path_texture['{:02d}'.format(obj_id)])
        texture_img_rgb=texture_img_bgr[:,:,2::-1]

    generate_codebook_imgs(path_model=path_model,
                           dir_imgs=dir_imgs, # RGB画像のパス
                           dir_edges=dir_edges, # Edge画像のパス
                           path_obj_bbs=path_obj_bbs, # 2d BBoxのパス
                           path_rot=path_rot, # サンプル?の回転行列のパス
                           render_dims=(720,540), # 画像のサイズ
                           cam_K=[1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1], # カメラ行列：cam_K=[f_x, 0, c_x(=w/2), 0, f_y, c_y(=h_2), 0, 0, 1]
                           depth_scale=1., # depth mapにこの値を掛けると，深度[mm]が求まる．今回はdepth mapの値=深度[mm]
                           texture_img=texture_img_rgb, # 今回はNone
                           start_end=[bid*batch_size,(bid+1)*batch_size]) # たぶん分割して実施する際のやつ．batch_size=50，bid=0の場合，start_end=[0, 50]
