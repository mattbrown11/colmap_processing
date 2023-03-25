# Pass the path to the Colmap data folder.

docker run -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    colmap/colmap:latest \
    colmap mapper \
        --database_path database.db \
        --image_path images0 \
        --output_path sparse \
        --Mapper.min_num_matches=15 \
        --Mapper.ignore_watermarks=0 \
        --Mapper.multiple_models=1 \
        --Mapper.max_num_models=50 \
        --Mapper.max_model_overlap=20 \
        --Mapper.min_model_size=10 \
        --Mapper.init_image_id1=-1 \
        --Mapper.init_image_id2=-1 \
        --Mapper.init_num_trials=200 \
        --Mapper.extract_colors=1 \
        --Mapper.num_threads=-1 \
        --Mapper.min_focal_length_ratio=0.10000000000000001 \
        --Mapper.max_focal_length_ratio=10 \
        --Mapper.max_extra_param=1 \
        --Mapper.ba_refine_focal_length=1 \
        --Mapper.ba_refine_principal_point=0 \
        --Mapper.ba_refine_extra_params=1 \
        --Mapper.ba_min_num_residuals_for_multi_threading=50000 \
        --Mapper.ba_local_num_images=6 \
        --Mapper.ba_local_max_num_iterations=25 \
        --Mapper.ba_global_use_pba=1 \
        --Mapper.ba_global_pba_gpu_index=-1 \
        --Mapper.ba_global_images_ratio=1.1000000000000001 \
        --Mapper.ba_global_points_ratio=1.1000000000000001 \
        --Mapper.ba_global_images_freq=500 \
        --Mapper.ba_global_points_freq=250000 \
        --Mapper.ba_global_max_num_iterations=50 \
        --Mapper.ba_global_max_refinements=5 \
        --Mapper.ba_global_max_refinement_change=0.00050000000000000001 \
        --Mapper.ba_local_max_refinements=2 \
        --Mapper.ba_local_max_refinement_change=0.001 \
        --Mapper.snapshot_images_freq=100 \
        --Mapper.fix_existing_images=0 \
        --Mapper.init_min_num_inliers=100 \
        --Mapper.init_max_error=4 \
        --Mapper.init_max_forward_motion=0.94999999999999996 \
        --Mapper.init_min_tri_angle=16 \
        --Mapper.init_max_reg_trials=2 \
        --Mapper.abs_pose_max_error=12 \
        --Mapper.abs_pose_min_num_inliers=30 \
        --Mapper.abs_pose_min_inlier_ratio=0.25 \
        --Mapper.filter_max_reproj_error=4 \
        --Mapper.filter_min_tri_angle=1.5 \
        --Mapper.max_reg_trials=3 \
        --Mapper.tri_max_transitivity=1 \
        --Mapper.tri_create_max_angle_error=2 \
        --Mapper.tri_continue_max_angle_error=2 \
        --Mapper.tri_merge_max_reproj_error=4 \
        --Mapper.tri_complete_max_reproj_error=4 \
        --Mapper.tri_complete_max_transitivity=5 \
        --Mapper.tri_re_max_angle_error=5 \
        --Mapper.tri_re_min_ratio=0.20000000000000001 \
        --Mapper.tri_re_max_trials=1 \
        --Mapper.tri_min_angle=1.5 \
        --Mapper.tri_ignore_two_view_tracks=1 \
        --Mapper.snapshot_path snapshots
