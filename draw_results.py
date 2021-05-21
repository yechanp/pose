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
                    help='output-directory', default="final_result")

parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
                    
parser.add_argument('--action_pred', dest='action_pred',
                    help='detection result file', default="results/action_pred")
                    
args = parser.parse_args()




def check_input():
    # for wecam
  

    # for video
    if len(args.video):
        args.outputpath = os.path.join(args.outputpath, 'video')
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    # if len(args.detfile):
        # args.outputpath = os.path.join(args.outputpath, 'video')
        # print('args.detfile',args.detfile)
        # if os.path.isfile(args.detfile):
            # detfile = args.detfile
            # return 'detfile', detfile
        # else:
            # raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        args.outputpath = os.path.join(args.outputpath, 'image')
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError
        
mode, input_source = check_input()        
        
if not os.path.isdir(args.outputpath):
    os.makedirs(args.outputpath)
if True:
    # print('im_name',im_name)
    # json_name = os.path.splitext(  im_name )[0] + '.json'
    # json_file_path = os.path.join(args.detfile,json_name)

    json_id_box  =  [[]]
    json_id_time =  [[]]
    
    video_id = os.path.splitext(os.path.split(args.video)[-1])[0]
    action_pred_list  = os.listdir(args.action_pred)
    action_pred_list  = [item for item in action_pred_list if video_id in item]
    print(video_id)
    print(action_pred_list)
    
    if len(action_pred_list)==0:
        raise ZeroDivisionError
    
   
            
            
    json_id_box  =  [[]]
    json_id_time =  [[]]
    with open(args.detfile, 'r',encoding='utf-8') as f:
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
                json_id_box.append( np.zeros( (len(det_res),4) )   )
                json_id_time.append([frame_now,frame_now])
            json_id_box[user_id][frame_now]  = res['box']
            if json_id_time[user_id][1] < frame_now:
                json_id_time[user_id][1] = frame_now
    
    rs  = [[]] * len( json_id_box)
    
    for action_pred in action_pred_list:
        id = int(action_pred.split('_')[-2])
        print('id',id)
        with open( os.path.join(args.action_pred,action_pred) ,'r') as f:
            r = f.readlines()
            r = [item.strip() for item in r]
            # print('r',r)
            rs[id] =r[1:]
    
  
    
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
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # print('fourcc',fourcc)
    
    
    
    def round_int(item,max_val=999999):
         return min(max(int(round(item)),0),max_val)
    
    # frame = cv2.putText(
               # frame, 
               # pred_label,
               # position,
               # cv2.FONT_HERSHEY_SIMPLEX ,
               # 0.5, #font_size
             # font_color, #font_color
               # 2 #font_stroke_width
           # )
    ext = os.path.splitext(args.video)[1]
    video_out_name = 'draw_{}_{}'.format(os.path.split(video_path)[-1],ext)
    video_out_name = os.path.join(args.outputpath,video_out_name)
    print('video_out_name',video_out_name)
    out = cv2.VideoWriter(video_out_name, fourcc, fps , (W,H))
        
    jump = 16
    for (i,frame) in enumerate(frames):
        for user_id in range(1,len(json_id_box)):
            start_frame , end_frame = json_id_time[user_id]
            if i>= start_frame and i<=end_frame:
                for r in rs[user_id]:
                    r = r.split(',')
                    # print('r',r)
                    if int(r[0]) <=i and int(r[1]) >= i:  
                        break
                box_info        =  np.array(json_id_box[user_id])
                box_idx         =  i
                if box_info[box_idx,0] > 1e-3:
                
                    y_0_min         =  round_int(box_info[box_idx,1])
                    x_0_min         =  round_int(box_info[box_idx,0])
                    y_1_max         =  round_int(box_info[box_idx,1]+box_info[box_idx,3])
                    x_1_max         =  round_int(box_info[box_idx,0]+box_info[box_idx,2])    
                
                
                pred_label = r[-1]    
                        
                # print('pred_label',pred_label)
                position   = (x_0_min,y_0_min+10)
                font_color = (255, 0, 0)
                
                frame = frame.copy()
                frame = cv2.putText(
                   frame, 
                   pred_label,
                   position,
                   cv2.FONT_HERSHEY_SIMPLEX ,
                   0.5, #font_size
                   font_color, #font_color
                   2) #font_stroke_width
                frame=cv2.rectangle(frame, (x_0_min, y_0_min), (x_1_max, y_1_max), font_color, 2)

            
        out.write(frame)
         
    # print('frame',frame.shape)
    # print('frame_clip',frame_clip.shape)
    out.release()
    
    
    # for user_id in range(1,len(json_id_box)):
        # # print('id' ,user_id)
        # box_info        = np.array(json_id_box[user_id])
        # y_0_min         =  round_int(box_info[:,1].min(),H)
        # x_0_min         =  round_int(box_info[:,0].min(),W)
        # y_1_max         =  round_int((box_info[:,1]+box_info[:,3]).max(),H)
        # x_1_max         =  round_int((box_info[:,0]+box_info[:,2]).max(),W)
        
        # box_bound = [y_0_min,x_0_min,y_1_max,x_1_max]
        # print('box_bound',box_bound)
       
        
        
        
        # max_box = box_info.max(axis=0)
        # print('max_box',max_box) 
        # pass
        # video_out_name = 'output_{}_{}.avi'.format(os.path.split(video_path)[-1],user_id)
        # print('video_out_name',video_out_name)
        # W_clip = x_1_max-x_0_min
        # H_clip = y_1_max-y_0_min
        
        # print('json_id_time',json_id_time[user_id])
        # start_frame , end_frame = json_id_time[user_id]

        
        
    
