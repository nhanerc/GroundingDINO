Follow the instructions in [README](README.md)

```bash
git checkout main

# This command will generate `pred.jpg` in `result` folder 
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i assets/demo1.jpg -o result -t "bear"

git checkout onnx
# Reinstall opnecv and install dependencies again
docker run -it --rm --gpus all -v `pwd`:/workspace -w /workspace nvcr.io/nvidia/pytorch:24.07-py3
# ONNX export => `dino.onnx` in `weights` folder
python onnx/export.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -o weights

# Outside the docker container
# This command will generate `output.jpg` in `result` folder
python onnx/inference.py -p weights/dino.onnx -i assets/demo1.jpg -o result -t "bear"
```