REM python scripts/demo_inference_json.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --video examples/video/ixCf7V5UV6o-Scene-005.mp4 --save_video --outdir ./results/res50_json/ --detfile results/res50/video/ixCf7V5UV6o-Scene-005.mp4.json --checkpoint pretrained_models/fast_res50_256x192.pth --pose_track


REM @echo OFF
for %%f in (examples\video\*) do ( 
ECHO %%f
python scripts/clip_video_junho.py --video %%f  --detfile results/res50/video/%%~nxf.json 
)