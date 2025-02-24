
singularity run --nv \
  -B $DATA_PATH/datasets:/data \
  -B $DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
  -B $DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -B $DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -B $DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -B $DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -B $DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -B $DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
  -B $DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw \
  -B $DATA_PATH/isaac-sim/documents:/root/Documents:rw \
  omnigibson_latest.sif
