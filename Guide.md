# LeWorldModel

环境
https://github.com/lucas-maes/le-wm/issues/48

```bash
# 构建
docker run -it \
  --name xwy_lewm \
  --init \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --shm-size=64G \
  --network host \
  -v /home/xuewenyao/code:/home/xuewenyao/code \
  -v /home/models:/home/models \
  -v /home/results:/home/results \
  -v /home/datasets:/home/datasets \
  -v /home/datasets_v2:/home/datasets_v2 \
  docker.1ms.run/pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

# 初始化
docker exec -u 0 -it -w /home/xuewenyao/code/LeWM xwy_lewm /bin/bash

# python 换源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set install.trusted-host mirrors.aliyun.com
pip config list
pip install stable-worldmodel[train,env]

apt-get update
apt-get install -y libxcb1 libx11-6 libxext6 libxrender1 libgl1 libglib2.0-0 libsm6 libgomp1
pip install -U "huggingface_hub[cli]"
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc 

# (1) 数据集: https://huggingface.co/datasets/quentinll/lewm-pusht
hf download quentinll/lewm-pusht --repo-type dataset --local-dir "/home/datasets_v2/LeWM/checkpoint/"
# (2) 检查点：https://huggingface.co/quentinll/lewm-pusht
hf download quentinll/lewm-pusht --repo-type model --local-dir "/home/datasets_v2/LeWM/checkpoint/lewm_pusht/"
zstd -d /home/datasets_v2/LeWM/checkpoint/pusht_expert_train.h5.zst -o /home/datasets_v2/LeWM/checkpoint/pusht_expert_train.h5

export STABLEWM_HOME=/home/datasets_v2/LeWM/checkpoint
python eval.py --config-name=pusht.yaml policy=lewm_pusht


```
