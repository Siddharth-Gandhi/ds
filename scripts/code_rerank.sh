#!/bin/bash
#SBATCH --job-name=cr
#SBATCH --output=logs/code_rerank/output_%A.log
#SBATCH --partition=gpu
#SBATCH --exclude=boston-2-25,boston-2-27,boston-2-29,boston-2-31
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=1


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "On host $(hostname)"
nvidia-smi

# repo_path="2_7/apache_spark"
# repo_path="2_7/apache_kafka"
# repo_path="2_7/facebook_react"
# repo_path="2_8/angular_angular"
# repo_path="2_8/django_django"
repo_path="smalldata/ftr"


index_path="${repo_path}/index_commit_tokenized"
k=1000 # initial ranker depth
n=100 # number of samples to evaluate on
# no_bm25=False # whether to use bm25 or not

model_path="microsoft/codebert-base"
# model_path="microsoft/graphcodebert-base"


# overwrite_cache=False # whether to overwrite the cache
batch_size=32 # batch size for inference
num_epochs=10 # number of epochs to train
learning_rate=5e-5 # learning rate for training
num_positives=10 # number of positive samples per query
num_negatives=10 # number of negative samples per querys
train_depth=1000 # depth to go while generating training data
num_workers=8 # number of workers for dataloader
train_commits=1000 # number of commits to train on (train + val)
psg_cnt=25 # number of commits to use for psg generation
psg_len=250
psg_stride=200
aggregation_strategy="sump" # aggregation strategy for bert reranker
rerank_depth=100 # depth to go while reranking
openai_model="gpt4" # openai model to use
# bert_best_model="${repo_path}/models/microsoft_codebert-base_model_output/best_model"
bert_best_model="2_7/facebook_react/models/microsoft_codebert-base_bertrr_gpt_train/best_model"
# best_model_path="data/combined/best_model"

repo_paths=(
    "2_7/apache_spark"
    "2_7/apache_kafka"
    "2_7/facebook_react"
    "2_8/angular_angular"
    "2_8/django_django"
    # "2_8/pytorch_pytorch"
    # "2_7/pandas-dev_pandas"
    # "2_7/julialang_julia"
    # "2_7/ruby_ruby"
    # "2_8/ansible_ansible"
    # "2_7/moby_moby"
    # "2_7/jupyter_notebook"
)

python -u src/CodeReranker.py \
    --repo_path $repo_path \
    --index_path $index_path \
    --k $k \
    --n $n \
    --model_path $model_path \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --num_positives $num_positives \
    --num_negatives $num_negatives \
    --train_depth $train_depth \
    --num_workers $num_workers \
    --train_commits $train_commits \
    --psg_cnt $psg_cnt \
    --psg_len $psg_len \
    --psg_stride $psg_stride \
    --use_gpu \
    --aggregation_strategy $aggregation_strategy \
    --rerank_depth $rerank_depth \
    --openai_model $openai_model \
    --bert_best_model $bert_best_model \
    --do_eval \
    --do_train \
    --ignore_gold_in_training
    # --eval_gold \
    # --sanity_check \
    # --overwrite_cache \



    # --do_combined \
    # --best_model_path $best_model_path \
    # --repo_paths "${repo_paths[@]}" \


echo "Job completed"









# repo_path="2_8/pytorch_pytorch" ???????
# repo_path="2_7/pandas-dev_pandas" ???????
# repo_path="2_7/julialang_julia"
# repo_path="2_7/ruby_ruby"
# repo_path="2_8/ansible_ansible"
# repo_path="2_7/moby_moby"
# repo_path="2_7/jupyter_notebook"