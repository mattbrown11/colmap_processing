## Mapper Parameters

**min_num_matches** (default=15)

**ignore_watermarks** (default=0)

**multiple_models** (default=1)

**max_num_models** (default=50)

**max_model_overlap** (default=20)

**min_model_size** (default=10)

**init_image_id1** (default=-1)

**init_image_id2** (default=-1)

**init_num_trials** (default=200)

**extract_colors** (default=1)

**num_threads** (default=-1)

**min_focal_length_ratio** (default=0.10000000000000001)

**max_focal_length_ratio** (default=10)

**max_extra_param** (default=1)

**ba_refine_focal_length** (default=1)

**ba_refine_principal_point** (default=0)

**ba_refine_extra_params** (default=1)

**ba_min_num_residuals_for_multi_threading** (default=50000)

**ba_local_num_images** (default=6)

**ba_local_max_num_iterations** (default=25)

**ba_global_use_pba** (default=0)

**ba_global_pba_gpu_index** (default=-1)

**ba_global_images_ratio** (default=1.1000000000000001)

**ba_global_points_ratio** (default=1.1000000000000001)

**ba_global_images_freq** (default=500)

**ba_global_points_freq** (default=250000)

**ba_global_max_num_iterations** (default=50)

**ba_global_max_refinements** (default=5)

**ba_global_max_refinement_change** (default=0.00050000000000000001)

**ba_local_max_refinements** (default=2)

**ba_local_max_refinement_change** (default=0.001)

**snapshot_images_freq** (default=0)

**fix_existing_images** (default=0)

**init_min_num_inliers** (default=100)

**init_max_error** (default=4)

**init_max_forward_motion** (default=0.94999999999999996)

**init_min_tri_angle** (default=16)

**init_max_reg_trials** (default=2)

**abs_pose_max_error** (default=12)

**abs_pose_min_num_inliers** (default=30)

**abs_pose_min_inlier_ratio** (default=0.25)

**filter_max_reproj_error** (default=4)

**filter_min_tri_angle** (default=1.5)

**max_reg_trials** (default=3)

Triangulation includes creation of new points, continuation of existing points, and merging of separate points if given image bridges tracks. Note that the given image must be registered and its pose must be set in the associated reconstruction.

**tri_max_transitivity** (default=1) : Maximum transitivity to search for correspondences.

**tri_create_max_angle_error** (default=2) : Maximum angular error (degrees) to create new triangulations.

**tri_continue_max_angle_error** (default=2) : Maximum angular error (degrees) to continue existing triangulations.

**tri_merge_max_reproj_error** (default=4) : Maximum reprojection error in pixels to merge triangulations.

**tri_complete_max_reproj_error** (default=4) : Maximum reprojection error to complete an existing triangulation.

**tri_complete_max_transitivity** (default=5) : Maximum transitivity for track completion.

**tri_re_max_angle_error** (default=5) : Maximum angular error (degrees) to re-triangulate under-reconstructed image pairs.

**tri_re_min_ratio** (default=0.20000000000000001) : Minimum ratio of common triangulations between an image pair over the number of correspondences between that image pair to be considered as under-reconstructed.

**tri_re_max_trials** (default=1) : Maximum number of trials to re-triangulate an image pair.

**tri_min_angle** (default=1.5) : Minimum pairwise triangulation angle for a stable triangulation.

**tri_ignore_two_view_tracks** (default=1) : Whether to ignore two-view tracks.