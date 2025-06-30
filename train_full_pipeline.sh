ITERS=30000
GLOBALARGS="-olambda_dssim=0.8"
FUNDATIONMODE="camera-shculling"
FUNDATIONARGS="
    -olambda_dssim=0.8
    -omercy_type='redundancy_opacity_opacity' \
    -oimportance_prune_interval=100"
ENHANCEMODE="camera-shculling"
ENHANCEARGS="
    -olambda_dssim=0.8
    -omercy_type='redundancy_opacity_opacity' \
    -oimportance_prune_interval=100 \
    -odensify_interval=500 \
    -oprune_interval=100"
BASRLINEITERS=30000
BASELINEMODE="camera-densify-prune-shculling"
BASELINEARGS="-omercy_type='redundancy_opacity_opacity'"
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
        -i $BASRLINEITERS \
        --mode $BASELINEMODE \
        -l output/$1/1x/point_cloud/iteration_$ITERS/point_cloud.ply \
        --load_camera output/$1/1x/cameras.json \
        $BASELINEARGS
}

pipeline truck
