## 1. Create Conda Environment
```sh
conda create -n interact-3d-scene-action python=3.10 -y
conda activate interact-3d-scene-action
```

## 2. Install PyTorch
```sh
conda install pytorch-gpu
```
## 3. Verify CUDA
```sh
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Remaining libraries
```sh
pip install transformers accelerate sentence-transformers
pip install fastapi uvicorn faiss-gpu
```

## 5. Optional but recommended
```sh
pip install bitsandbytes peft trl datasets
```

## 6. Run the fast api app
```sh
uvicorn main:app --host 127.0.0.1 --port 8000
```

The API endpoint sample
```sh
 http://127.0.0.1:8000/command?query=highlight me the front wheel from left&obj=front_door,rear_door,engine, wheel_front_left, wheel_front_right
 ```
expected response:
```sh
{"action":"highlight","object":"wheel_front_left"}
```