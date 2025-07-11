ITERS=30000
GLOBALARGS="-olambda_dssim=0.8 --with_scale_reg"
FUNDATIONMODE="shculling" # Do not optimize camera (load from reduced-3dgs, already trained)
FUNDATIONARGS="
    -omercy_type='redundancy_opacity_opacity' \
    -oimportance_prune_from_iter=1000 \
    -oimportance_prune_interval=100"
ENHANCEMODE="shculling"
ENHANCEARGS="
    -omercy_type='redundancy_opacity_opacity' \
    -oimportance_prune_from_iter=1000 \
    -oimportance_prune_interval=100 \
    -odensify_interval=500 \
    -oprune_interval=100"
# ENHANCEARGS="$ENHANCEARGS \
#     -oreset_fixed_opacity_to=0.95 \
#     -ofix_features_dc=False \
#     -ofix_features_rest=False \
#     -ofix_scaling=False \
#     -ofix_rotation=False \
#     -ofix_opacity=False"
BASRLINEITERS=30000
BASELINEMODE="camera-densify-prune-shculling"
BASELINEARGS="-omercy_type='redundancy_opacity_opacity' --with_scale_reg"
pipeline() {
    # Train the baseline model, regular reduced-3dgs
    python -m reduced_3dgs.train \
        -s data/$1 -d output/$1 \
        -i $BASRLINEITERS \
        --mode $BASELINEMODE \
        $BASELINEARGS
    # Train the fundation layer (8x), use the trained camera and scene from reduced-3dgs
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/8x \
        --rescale_factor 0.125 -i $ITERS \
        --mode $FUNDATIONMODE \
        --load_camera output/$1/cameras.json \
        $GLOBALARGS $FUNDATIONARGS
    # Train the next layer (4x), use the trained camera from reduced-3dgs and scene from 8x
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/4x \
        --rescale_factor 0.25 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/8x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/cameras.json \
        $GLOBALARGS $ENHANCEARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/2x \
        --rescale_factor 0.5 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/4x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/cameras.json \
        $GLOBALARGS $ENHANCEARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/1x \
        --rescale_factor 1.0 -i $ITERS \
        --mode $ENHANCEMODE \
        -l output/$1/2x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/cameras.json \
        $GLOBALARGS $ENHANCEARGS
}

pipeline truck
