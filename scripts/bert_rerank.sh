#!/bin/bash
#SBATCH --job-name=bert_rerank
#SBATCH --output=logs/bert_rerank/bert_rerank_output_%A.log
#SBATCH --partition=gpu
# SBATCH --exclude=boston-2-25,boston-2-27,boston-2-29,boston-2-31
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=30G
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
# repo_path="2_8/pytorch_pytorch" ???????
repo_path="2_8/django_django"
# repo_path="2_7/pandas-dev_pandas" ???????
# repo_path="2_7/julialang_julia"
# repo_path="2_7/ruby_ruby"
# repo_path="2_8/ansible_ansible"
# repo_path="2_7/moby_moby"
# repo_path="2_7/jupyter_notebook"
index_path="${repo_path}/index_commit_tokenized"
k=1000 # initial ranker depth
n=100 # number of samples to evaluate on
no_bm25=False # whether to use bm25 or not
model_path="microsoft/codebert-base"
overwrite_cache=False # whether to overwrite the cache
batch_size=32 # batch size for inference
num_epochs=10 # number of epochs to train
learning_rate=5e-5 # learning rate for training
num_positives=10 # number of positive samples per query
num_negatives=10 # number of negative samples per querys
train_depth=1000 # depth to go while generating training data
num_workers=8 # number of workers for dataloader
train_commits=1500 # number of commits to train on (train + val)
psg_cnt=5 # number of commits to use for psg generation
aggregation_strategy="sump" # aggregation strategy for bert reranker
use_gpu=True # whether to use gpu or not
rerank_depth=250 # depth to go while reranking
do_train=True # whether to train or not
do_eval=True # whether to evaluate or not
openai_model="gpt4" # openai model to use

python -u src/BERTReranker_v4.py \
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
    --aggregation_strategy $aggregation_strategy \
    --rerank_depth $rerank_depth \
    --openai_model $openai_model \
    --use_gpu \
    --eval_gold \
    # --sanity_check \
    # --do_eval \
    # --do_train \
    # --eval_before_training \
    # --no_bm25 \
    # --debug \
    # --overwrite_cache \

echo "Job completed"