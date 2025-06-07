GLOBALARGS="-olambda_dssim=0.8"
pipeline() {
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/8x \
        --rescale_factor 0.125 -i $2 \
        --mode shculling \
        $GLOBALARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/4x \
        --rescale_factor 0.25 -i $2 \
        --mode camera-shculling \
        -l output/$1/8x/point_cloud/iteration_$2/point_cloud.ply \
        --load_camera output/$1/8x/cameras.json \
        $GLOBALARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/2x \
        --rescale_factor 0.5 -i $2 \
        --mode camera-shculling \
        -l output/$1/4x/point_cloud/iteration_$2/point_cloud.ply \
        --load_camera output/$1/4x/cameras.json \
        $GLOBALARGS
    python -m lapisgs.train_reduced \
        -s data/$1 -d output/$1/1x \
        --rescale_factor 1.0 -i $2 \
        --mode camera-shculling \
        -l output/$1/2x/point_cloud/iteration_$2/point_cloud.ply \
        --load_camera output/$1/2x/cameras.json \
        $GLOBALARGS
}

pipeline truck 30000
