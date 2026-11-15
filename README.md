# AI-based 5G Scheduler (NVIDIA Aerial)

- PPO-based RL scheduler
- GPU accelerated (CUDA)
- Dynamic resource allocation

## Results

- Stable reward convergence
- Real-time inference capability

## To Run the Model

## Terminal 1

cd ~/5g_ai_scheduler
source venv/bin/activate

docker run --gpus all -it --rm \
-p8000:8000 \
-v $(pwd)/model_repository:/models \
nvcr.io/nvidia/tritonserver:24.01-py3 \
tritonserver --model-repository=/models

## If Triton Ports are engaged

docker ps
docker stop <container id>

# or

docker rm -f $(docker ps -aq)

## Terminal 2

cd ~/5g_ai_scheduler
source venv/bin/activate

streamlit run dashboard.py
