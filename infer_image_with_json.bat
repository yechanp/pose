@echo OFF
python scripts/demo_inference_json.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir ./examples/image/  --save_img --outdir ./results/res50_json/ --detfile results/res50/image/
