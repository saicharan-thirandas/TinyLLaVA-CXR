python3 tinyllava/eval/model_vqa_loader_batch.py --model-path /data/tinyllava-checkpoint/archive/con-and-llm-full-reports/checkpoint-3400/  --question-file /data/annotations/mimic-cxr-jpg/jsons/mimic_conversation_test_images_questions.jsonl --conv-mode pretrain --answers-file answer/answer.jsonl  --image-folder /data/mount2/mimic-cxr-jpg-2.1.0.physionet.org/ --batch-size 12