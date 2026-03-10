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