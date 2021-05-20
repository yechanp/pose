@echo OFF
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir ./examples/image/  --save_img --outdir ./results/res50/

set file=D:\Personal Carrier\Pose\AlphaPose_junho\examples\image\1.jpg
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --image file --save_img --outdir ./results/res50 