CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /home/share/xutianjiao/code/pretrained/glm-4-9b-chat \
    --served-model-name glm4_9b \
    --port 8880 \
    --max-model-len=40000 \
    --trust-remote-code \
    --gpu-memory-utilization=0.8