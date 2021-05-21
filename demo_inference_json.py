"""Script for single-gpu/multi-gpu demo."""
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

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
# from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter_writeonly as DataWriter

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

# sys.argv = ['scripts/demo_inference.py', '--cfg', 'configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml', '--checkpoint', 'pretrained_models/fast_res50_256x192.pth', '--indir', './examples/image/', '--save_img', '--outdir', './results/res50']
# sys.argv = ['scripts/demo_inference.py', '--cfg', 'configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml', '--checkpoint', 'pretrained_models/fast_res50_256x192.pth', '--video', './examples/video/vZvY_pPfLKQ-Scene-017.mp4', '--outdir', './examples/res/res50/']
args = parser.parse_args()
cfg = update_config(args.cfg)


if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

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


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1

def vis_frame_fast(frame, hm, opt, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = len(hm)//3
    # if len(im_res['result']) > 0:
    	# kp_num = len(im_res['result'][0]['keypoints'])
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
            (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
            (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
            (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
            (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
            (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
            (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
            (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
            (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        # part_line = {}
        # kp_preds = human['keypoints']
        # kp_scores = human['kp_score']
        # if kp_num == 17:
            # kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            # kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        # if opt.pose_track or opt.tracking:
            # color = get_color_fast(int(abs(human['idx'])))
        # else:
            # color = BLUE

        # Draw bboxes
        if opt.showbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax
            else:
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)
            
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if opt.tracking:
                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        # Draw keypoints
        vis_thres = 0.05 if kp_num == 136 else 0.4
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255,255,255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if opt.tracking:
                        cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255,255,255), 1)  

    return img


if __name__ == "__main__":
    mode, input_source = check_input()
    
    if mode == 'video':
        with open(args.detfile, 'r',encoding='utf-8') as f:
            det_res = json.load(f)
        res_frame = []
        frame = 0 
        print('len(det_res)',len(det_res))
        for idx in range(0, len(det_res)):
            res       = det_res[idx]
            frame_now = int( res['image_id'].split('.')[0] )
            # print('frame_now',frame_now)
            while(len(res_frame) < frame_now+1):
                res_frame.append([])
            res_frame[frame_now].append(res)
    elif mode == 'image':
        res_frame = []
        frame_now = 0
        # for root, dirs, files in os.walk(args.detfile):
            # json_names = files
            # json_names = [item for item in json_names if '.json' in item] 
            # json_names = natsort.natsorted(json_names)
        except_idx = [] 
        for (idx, im_name) in enumerate(input_source):
            json_name = os.path.splitext(im_name)[0] + '.json'
            json_file_path = os.path.join(args.detfile,json_name)
            if not os.path.isfile( json_file_path ):
                print('There is no json file corresponding to image file : ' , json_file_path )
                except_idx.append(idx)
                continue
            with open(json_file_path , 'r' , encoding='utf-8') as f:
                print('loading json :',json_file_path)
                det_res = json.load(f)
                res_frame.append([])
                for res_idx in range(len(det_res)):
                    res     = det_res[res_idx]
                    res_frame[-1].append(res)
        input_source = [ input_source[idx] for idx in range(len(input_source)) if idx not in except_idx]
        print('input_source',input_source)
        
        
    
   
    # print('res_frame[0]',res_frame[0])
    # print('len(res_frame)',len(res_frame))
    # raise ZeroDivisionError
    
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()
    print('mode' , mode)

    # Load pose model
    # pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    # print('Loading pose model from %s...' % (args.checkpoint,))
    print('Not Loading pose model' )
    # pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    # if args.pose_track:
        # tracker = Tracker(tcfg, args)
    # if len(args.gpus) > 1:
        # pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    # else:
        # pose_model.to(args.device)
    # pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
    else:
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    # try:
    print('data_len',data_len)
    # raise ZeroDivisionError
    
                
    

    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
            res_now =  res_frame[i]
            hm  = []
            ids = []
            for k in range(len(res_now)):
                hm.append(res_now[k]['keypoints'])
                ids.append(res_now[k]['idx'])
            hm  = np.array(hm)
            hm  = torch.from_numpy(hm)
            # print('hm.shape',hm.shape)
            # continue
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name)
                continue
            if args.profile:
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
            # Pose Estimation (model inference하는 부분)
            # inps = inps.to(args.device)
            # datalen = inps.size(0)
            # leftover = 0
            # if (datalen) % batchSize:
                # leftover = 1
            # num_batches = datalen // batchSize + leftover
            # hm = []
            # for j in range(num_batches):
                # inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                # if args.flip:
                    # inps_j = torch.cat((inps_j, flip(inps_j)))
                # hm_j = pose_model(inps_j)
                # if args.flip:
                    # hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                    # hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                # hm.append(hm_j)
            # hm = torch.cat(hm)
            # if args.profile:
                # ckpt_time, pose_time = getTime(ckpt_time)
                # runtime_profile['pt'].append(pose_time)
            # if args.pose_track:
                # boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
            # hm = hm.cpu()
            writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
            if args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )
    print_finish_info()
    while(writer.running()):
        time.sleep(1)
        print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
    writer.stop()
    det_loader.stop()
    # except Exception as e:
        # print(repr(e))
        # print('An error as above occurs when processing the images, please check it')
        # pass
    # except KeyboardInterrupt:
        # print_finish_info()
        # # Thread won't be killed when press Ctrl+C
        # if args.sp:
            # det_loader.terminate()
            # while(writer.running()):
                # time.sleep(1)
                # print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            # writer.stop()
        # else:
            # # subprocesses are killed, manually clear queues

            # det_loader.terminate()
            # writer.terminate()
            # writer.clear_queues()
            # det_loader.clear_queues()

