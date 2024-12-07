# Next 3 splits on GPU 1
for num in {4..6}; do
  CUDA_VISIBLE_DEVICES=1 nohup python3 bin/cxas_segment_bulk_file_input.py -i /home/thiras3/workspace/TinyLLaVA-CXR/annotations/mimic-cxr-jpg/split_image_files/image_filepaths_split_${num}.txt > nohup_${num} 2>&1 &
  sleep 10
done



# First 3 splits on GPU 0
for num in {1..3}; do
  CUDA_VISIBLE_DEVICES=0 nohup python3 bin/cxas_segment_bulk_file_input.py -i /home/thiras3/workspace/TinyLLaVA-CXR/annotations/mimic-cxr-jpg/split_image_files/image_filepaths_split_${num}.txt > nohup_${num} 2>&1 &
  sleep 10
done

