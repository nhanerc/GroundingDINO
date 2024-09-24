```bash
# 1. Clone the project
git clone https://github.com/nhanerc/GroundingDINO.git
cd GroundingDINO
git checkout main

# 2. Use docker container
docker run -it --rm --gpus all -v `pwd`:/workspace -w /workspace nvcr.io/nvidia/pytorch:24.07-py3
pip install -e .
pip uninstall opencv opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.8.0.74 onnxruntime

# 2. This command will generate `pred.jpg` in `result` folder
mkdir -p weights && wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i assets/demo1.jpg -o result -t "bear"

# 3. Open another terminal (outside the docker container): git checkout onnx

# 4. Go back to the docker container, ONNX export => `dino.onnx` in `weights` folder
python onnx/export.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -o weights

# 5. This command will generate `output.jpg` in `result` folder
python onnx/inference.py -p weights/dino.onnx -i assets/demo1.jpg -o result --box_threshold 0.35 -t "bear"
```