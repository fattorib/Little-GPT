# Download inference checkpoints
python download_checkpoints.py --model-size 127 
python download_checkpoints.py --model-size 303 

# Evaluate 127* with ctx varying from 1024 to 8192
python benchmark.py --dataset text8,enwik8,WikiText2 --model base* --type GPT2 --eval-ctx 1024,2048,3072,4096,8192 
# Evaluate 303* with ctx varying from 1024 to 4096 (8192 OOMs on 3090)
python benchmark.py --dataset text8,enwik8,WikiText2 --model medium* --type GPT2 --eval-ctx 1024,2048,3072,4096


python benchmark.py --dataset WikiText2 --model medium* --type GPT2 --eval-ctx 2048