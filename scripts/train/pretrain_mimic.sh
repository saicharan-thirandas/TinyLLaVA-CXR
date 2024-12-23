PRETRAIN_DATA_PATH="/home/thiras3/workspace/TinyLLaVA-CXR/annotations/mimic_reports_train_images.json" #finetune annotation file path
PRETRAIN_IMAGE_PATH="/home/thiras3/workspace/s3bucket/mimic-cxr-jpg-2.1.0.physionet.org" #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm


bash scripts/train/pretrain.sh "$PRETRAIN_DATA_PATH" "$PRETRAIN_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
