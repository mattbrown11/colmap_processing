# Pass the path to the Colmap data folder.

docker pull colmap/colmap:latest

xhost +local:docker

docker run --rm -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    --name colmap_gui \
    colmap/colmap:latest \
    colmap gui
