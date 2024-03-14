#!/bin/bash
#SBATCH --job-name=be_reg_fb
#SBATCH --output=logs/bert_rerank/output_%A.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "On host $(hostname)"
nvidia-smi
export TOKENIZERS_PARALLELISM=true


eval_folder="combined_bert_reg_eval"
notes="combined train with eval"


# data_path="2_7/apache_spark"
# data_path="2_7/apache_kafka"
data_path="data/2_7/facebook_react"
# data_path="2_8/angular_angular"
# data_path="2_8/django_django"
# data_path="2_7/julialang_julia"
# data_path="2_7/ruby_ruby"
# data_path="data/2_8/pytorch_pytorch"
# data_path="2_9/huggingface_transformers"
# data_path="2_9/redis_redis"

# data_path="smalldata/ftr"

train_mode="regression"

repo_paths=(
    "data/2_7/apache_spark"
    "data/2_7/apache_kafka"
    "data/2_7/facebook_react"
    "data/2_8/angular_angular"
    "data/2_8/django_django"
    "data/2_8/pytorch_pytorch"
    "data/2_7/julialang_julia"
    "data/2_7/ruby_ruby"
    "data/2_9/huggingface_transformers"
    "data/2_9/redis_redis"
)



index_path="${data_path}/index_commit_tokenized"
k=1000 # initial ranker depth
n=100 # number of samples to evaluate on
# no_bm25=False # whether to use bm25 or not

model_path="microsoft/codebert-base"
# model_path="microsoft/graphcodebert-base"


# overwrite_cache=False # whether to overwrite the cache
batch_size=32 # batch size for inference
num_epochs=20 # number of epochs to train
learning_rate=5e-5 # learning rate for training
num_positives=10 # number of positive samples per query
num_negatives=10 # number of negative samples per querys
train_depth=1000 # depth to go while generating training data
num_workers=8 # number of workers for dataloader
train_commits=1000 # number of commits to train on (train + val)
psg_cnt=5 # number of commits to use for psg generation
aggregation_strategy="sump" # aggregation strategy for bert reranker
# use_gpu=True # whether to use gpu or not
rerank_depth=250 # depth to go while reranking
# do_train=True # whether to train or not
# do_eval=True # whether to evaluate or not
openai_model="gpt4" # openai model to use
# best_model_path="data/combined_commit_train/best_model"
best_model_path="/home/ssg2/ssg2/ds/data/combined_gpt_train/combined_bce_train/best_model"


python -u src/BERTReranker_v4.py \
    --data_path $data_path \
    --index_path $index_path \
    --k $k \
    --n $n \
    --model_path $model_path \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --run_name $eval_folder \
    --notes "$notes" \
    --num_positives $num_positives \
    --num_negatives $num_negatives \
    --train_depth $train_depth \
    --num_workers $num_workers \
    --train_commits $train_commits \
    --psg_cnt $psg_cnt \
    --use_gpu \
    --aggregation_strategy $aggregation_strategy \
    --rerank_depth $rerank_depth \
    --openai_model $openai_model \
    --eval_folder $eval_folder \
    --repo_paths "${repo_paths[@]}" \
    --eval_gold \
    --no_bm25 \
    --use_gpt_train \
    --do_eval \
    --do_train \
    # --train_mode $train_mode \


    # --do_combined_train \
    # --sanity_check \
    # --best_model_path $best_model_path \
    # --overwrite_cache \
    # --ignore_gold_in_training \


    # --debug \

echo "Job completed"









# data_path="2_7/pandas-dev_pandas" ???????
# data_path="2_8/ansible_ansible"
# data_path="2_7/moby_moby"
# data_path="2_7/jupyter_notebook"

# (
#     "apache_spark"
#     "apache_kafka"
#     "facebook_react"
#     "angular_angular"
#     "django_django"
#     "pytorch_pytorch"
#     "julialang_julia"
#     "ruby_ruby"
#     "huggingface_transformers"
#     "redis_redis"
# )