protocol_file="/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/protocol_train_dev.txt"
output_dir="/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/preprocessed_data"

python preprocess_all_data.py \
      --protocol_file $protocol_file \
      --output_dir $output_dir \
      --num_workers 8