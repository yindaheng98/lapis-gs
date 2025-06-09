ITERS=10000
GLOBALARGS="
    -olambda_dssim=0.8
    --with_scale_reg \
    -oscale_reg_thr_scale=0.2 \
    -odensify_percent_too_big=0.15 \
    -odepth_l1_weight_init=1.0 \
    -odepth_l1_weight_final=1.0 \
    -odepth_from_iter=4000 \
    -odepth_local_relative_kernel_radius=32 \
    -odepth_local_relative_stride=16 \
    -odepth_resize=577 \
    -omercy_type='redundancy_opacity_opacity' \
    -oposition_lr_max_steps=10000 \
    -ocull_at_steps=[9000] \
    -oscale_reg_from_iter=500 \
    -odepth_l1_weight_max_steps=10000 \
    -oimportance_prune_from_iter=2000 \
    -oimportance_prune_until_iter=8500 \
    -oimportance_prune_interval=100 \
    -oopacity_reset_from_iter=3000 \
    -oopacity_reset_until_iter=7000 \
    -oopacity_reset_interval=500"
FUNDATIONMODE="shculling"
FUNDATIONARGS="
    -odensify_from_iter=1000 \
    -odensify_until_iter=7800 \
    -odensify_interval=100 \
    -oprune_from_iter=2000 \
    -oprune_until_iter=7500 \
    -oprune_interval=500"
ENHANCEMODE="camera-shculling"
ENHANCEARGS="
    -odensify_from_iter=1000 \
    -odensify_until_iter=7800 \
    -odensify_interval=500 \
    -oprune_from_iter=2000 \
    -oprune_until_iter=7500 \
    -oprune_interval=100 \
    -ocamera_position_lr_max_steps=10000 \
    -ocamera_rotation_lr_max_steps=10000 \
    -ocamera_exposure_lr_max_steps=10000"
BASELINEMODE="camera-densify-prune-shculling"
BASELINEARGS="
    -odensify_from_iter=1000 \
    -odensify_until_iter=7800 \
    -odensify_interval=500 \
    -oprune_from_iter=2000 \
    -oprune_until_iter=7500 \
    -oprune_interval=100 \
    -ocamera_position_lr_max_steps=10000 \
    -ocamera_rotation_lr_max_steps=10000 \
    -ocamera_exposure_lr_max_steps=10000"
pipeline() {
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/8x \
        --rescale_factor 0.125 -i $ITERS \
        --mode $FUNDATIONMODE \
        $GLOBALARGS $FUNDATIONARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/4x \
        --rescale_factor 0.25 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/8x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/8x/cameras.json \
        $GLOBALARGS $ENHANCEARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/2x \
        --rescale_factor 0.5 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/4x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/4x/cameras.json \
        $GLOBALARGS $ENHANCEARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/1x \
        --rescale_factor 1.0 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/2x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/2x/cameras.json \
        $GLOBALARGS $ENHANCEARGS
    python -m reduced_3dgs.train \
        -s data/$1 -d output/$1 \
        -i $ITERS \
        --mode $BASELINEMODE \
        $GLOBALARGS $BASELINEARGS
}

pipeline truck
