docker run --name siqi_slime_opsd --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/disk1_from_server2/siqizhu4/opsd_slime:/root/slime_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/ray_tmp:/tmp/ray_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/checkpoints:/root/checkpoints_siqi \
  -w /root/slime_siqi \
  -it slimerl/slime:latest /bin/bash

docker exec -it siqi_slime_opsd bash

bash start.ps1

apt install iproute2 -y

export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0