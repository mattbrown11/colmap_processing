# Pass the path to the Colmap data folder.

docker pull colmap/colmap:latest

docker run -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    colmap/colmap:latest \
    colmap patch_match_stereo \
        --workspace_path dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true
        --pmvs_option_name option-all \
        --PatchMatchStereo.max_image_size -1 \
        --PatchMatchStereo.gpu_index -1 \
        --PatchMatchStereo.depth_min -1 \
        --PatchMatchStereo.depth_max -1 \
        --PatchMatchStereo.window_radius 25 \
        --PatchMatchStereo.window_step 1 \
        --PatchMatchStereo.sigma_spatial -1 \
        --PatchMatchStereo.sigma_color 0.20000000298023224
        --PatchMatchStereo.num_samples 15 \
        --PatchMatchStereo.ncc_sigma 0.60000002384185791 \
        --PatchMatchStereo.min_triangulation_angle 1 \
        --PatchMatchStereo.incident_angle_sigma 0.89999997615814209 \
        --PatchMatchStereo.num_iterations 5 \
        --PatchMatchStereo.geom_consistency 1 \
        --PatchMatchStereo.geom_consistency_regularizer 0.30000001192092896 \
        --PatchMatchStereo.geom_consistency_max_cost 3 \
        --PatchMatchStereo.filter 1 \
        --PatchMatchStereo.filter_min_ncc 0.10000000149011612 \
        --PatchMatchStereo.filter_min_triangulation_angle 3 \
        --PatchMatchStereo.filter_min_num_consistent 2 \
        --PatchMatchStereo.filter_geom_consistency_max_cost 1 \
        --PatchMatchStereo.cache_size 128 \
        --PatchMatchStereo.write_consistency_graph 0
