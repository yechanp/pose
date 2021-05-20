@echo OFF
for %%f in (examples\video\*) do ( 
ECHO %%f
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video %%f --save_video --outdir ./results/res50/ --pose_track
)


REM python scripts/demo_inference_json.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --video examples/video/ixCf7V5UV6o-Scene-005.mp4 --save_video --outdir ./results/res50_json/ --detfile results/res50/video/ixCf7V5UV6o-Scene-005.mp4.json --checkpoint pretrained_models/fast_res50_256x192.pth --pose_track