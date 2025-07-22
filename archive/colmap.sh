# The project folder must contain a folder "images" with all the images.
DATASET_PATH=./tmp/colmap2

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/img

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/img \
    --output_path $DATASET_PATH/sparse


# DENSE

# mkdir $DATASET_PATH/dense

# colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \
#     --max_image_size 2000

# colmap patch_match_stereo \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# colmap stereo_fusion \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path $DATASET_PATH/dense/fused.ply

# colmap poisson_mesher \
#     --input_path $DATASET_PATH/dense/fused.ply \
#     --output_path $DATASET_PATH/dense/meshed-poisson.ply

# colmap delaunay_mesher \
#     --input_path $DATASET_PATH/dense \
#     --output_path $DATASET_PATH/dense/meshed-delaunay.ply



## OTHER COLMAP COMMANDS

# CAMERA_PARAMS="601.4786987304688,167.90625,298.5"

# Run COLMAP pipeline
# colmap feature_extractor \
#   --database_path $DB \
#   --image_path $IMG \
#   --ImageReader.camera_model SIMPLE_PINHOLE \
#   --ImageReader.single_camera 1 \
#   --ImageReader.camera_params "$CAMERA_PARAMS"

# colmap exhaustive_matcher \
#   --database_path $DB

# mkdir -p $OUT

# colmap mapper \
#   --database_path $DB \
#   --image_path $IMG \
#   --output_path $OUT
