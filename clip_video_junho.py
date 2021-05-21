import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort
import json
import cv2


parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="action_preprocess")

parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
# sys.argv = ['scripts/clip_video.py', '--video', './examples/video/p-15zlQgsGg-Scene-025_0_.mp4']                    
args = parser.parse_args()
detfile = './results/res50/video/' + os.path.split(args.video)[-1] + '.json'
# detfile = args.detfile

def check_input():
    # for wecam
  

    # for video
    if len(args.video):
        # args.outputpath = os.path.join(args.outputpath, 'video')
        os.makedirs(args.outputpath, exist_ok=True)
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    # if len(detfile):
        # args.outputpath = os.path.join(args.outputpath, 'video')
        # print('detfile',detfile)
        # if os.path.isfile(detfile):
            # detfile = detfile
            # return 'detfile', detfile
        # else:
            # raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    # if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
    #     args.outputpath = os.path.join(args.outputpath, 'image')
    #     inputpath = args.inputpath
    #     inputlist = args.inputlist
    #     inputimg = args.inputimg

    #     if len(inputlist):
    #         im_names = open(inputlist, 'r').readlines()
    #     elif len(inputpath) and inputpath != '/':
    #         for root, dirs, files in os.walk(inputpath):
    #             im_names = files
    #         im_names = natsort.natsorted(im_names)
    #     elif len(inputimg):
    #         args.inputpath = os.path.split(inputimg)[0]
    #         im_names = [os.path.split(inputimg)[1]]

    #     return 'image', im_names

    else:
        raise NotImplementedError
        
mode, input_source = check_input()        
        

if True:
    # print('im_name',im_name)
    # json_name = os.path.splitext(  im_name )[0] + '.json'
    # json_file_path = os.path.join(detfile,json_name)

    json_id_box  =  [[]]
    json_id_time =  [[]]
    json_id_bframe = [[]]
    with open(detfile, 'r',encoding='utf-8') as f:
        det_res = json.load(f)
        res_frame = []
        frame = 0 
        # print('len(det_res)',len(det_res))
        for idx in range(0, len(det_res)):
            res       = det_res[idx]
            frame_now = int( res['image_id'].split('.')[0] )
            user_id   = res['idx']
            # print('user_id',user_id)
            if user_id >= len(json_id_box):
                json_id_box.append([])
                json_id_bframe.append([])
                json_id_time.append([frame_now,frame_now])
            json_id_box[user_id].append(res['box'])
            json_id_bframe[user_id].append(frame_now)
            if json_id_time[user_id][1] < frame_now:
                json_id_time[user_id][1] = frame_now
        # json_id_time[user_id] = [start_frame,end_frame]
                
            
            
            # # print('frame_now',frame_now)
            # while(len(res_frame) < frame_now+1):
                # res_frame.append([])
            # res_frame[frame_now].append(res)
            
  
    
    video_path = args.video
    # print('video_path',video_path , os.path.isfile(video_path))
    cap      = cv2.VideoCapture(video_path)
    fps      = int(round(cap.get(cv2.CAP_PROP_FPS)))

    frames = []
    while(cap.isOpened()):
        ret,frame = cap.read()
        # if not ret:
            # break
        if frame is None:
            break
        frames.append(frame)
        

    frames  = np.array(frames)
    cap.release()
    video_shape = frames.shape
    F , H, W ,C = video_shape
    # print('video_shape',video_shape)
    
    # assert 1==2
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # print('fourcc',fourcc)
    
    def round_int(item,max_val=999999):
         return min(max(int(round(item)),0),max_val)
         
    broad_ratio = 1.05 #1.05
    
    for user_id in range(1,len(json_id_box)):
        # print('id' ,user_id)
        box_info        =  np.array(json_id_box[user_id])
        frame_info      =  np.array(json_id_bframe[user_id])
        
        box_info[:,0]   = box_info[:,0]-( box_info[:,2] * (broad_ratio-1) )
        box_info[:,1]   = box_info[:,1]-( box_info[:,3] * (broad_ratio-1) )
        box_info[:,2]   = box_info[:,2]*broad_ratio
        box_info[:,3]   = box_info[:,3]*broad_ratio
        
        x_0_min         =  round_int(box_info[:,0].min(),W)
        y_0_min         =  round_int(box_info[:,1].min(),H)
        x_1_max         =  round_int((box_info[:,0]+box_info[:,2]).max(),W)
        y_1_max         =  round_int((box_info[:,1]+box_info[:,3]).max(),H)
        x_len_max       =  round_int((box_info[:,2]).max(),W)
        y_len_max       =  round_int((box_info[:,3]).max(),H)        
        box_bound = [y_0_min,x_0_min,y_1_max,x_1_max]
        print('box_bound',box_bound)
        
        
        
        # max_box = box_info.max(axis=0)
        # print('max_box',max_box) 
        pass
        # video_out_name = 'output_{}_{}.avi'.format(os.path.split(video_path)[-1],user_id)
        ext = os.path.splitext(args.video)[1]
        video_out_name = 'output_{}_{}{}'.format(os.path.split(video_path)[-1],user_id,ext)

        
        print('video_out_name',os.path.join(args.outputpath,video_out_name))
        
       
        
        W_clip = x_len_max
        H_clip = y_len_max
        
        # W_clip = x_1_max-x_0_min
        # H_clip = y_1_max-y_0_min
        
        print('json_id_time',json_id_time[user_id])
        start_frame , end_frame = json_id_time[user_id]

        out = cv2.VideoWriter(os.path.join(args.outputpath,video_out_name), fourcc, fps , (W_clip,H_clip))

        box_cnt = 0
        for (i,frame) in enumerate(frames):
            if i>= start_frame and i<=end_frame:
                frame_y_min = round_int(box_info[box_cnt,1])
                frame_y_max = frame_y_min + y_len_max
                frame_x_min = round_int(box_info[box_cnt,0])
                frame_x_max = frame_x_min + x_len_max
                pad_frame_clip = np.zeros((y_len_max,x_len_max,3)).astype('uint8')
                
                
                frame_clip = np.array(frame[frame_y_min:frame_y_max, frame_x_min:frame_x_max, :])
                pad_frame_clip[:frame_clip.shape[0],:frame_clip.shape[1],:frame_clip.shape[2]] = frame_clip
                out.write(pad_frame_clip)
                if i in frame_info:
                    box_cnt += 1
                # frame_clip = np.array(frame[y_0_min:y_1_max,x_0_min:x_1_max,:])
                # out.write(frame_clip)
                
             
        print('frame',frame.shape)
        print('frame_clip',frame_clip.shape)
        out.release()
        
    
