#!/bin/bash
#SBATCH --job-name=bce
#SBATCH --output=logs/code_rerank/output_%A.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --nodelist=boston-1-7

# Activate the conda environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "On host $(hostname)"
nvidia-smi
export TOKENIZERS_PARALLELISM=true


eval_folder="bce"
notes="bce"
# triplet_mode="parse_functions"
triplet_mode="sliding_window"
# triplet_mode="diff_content"
# triplet_mode="diff_subsplit"



# repo_path="2_8/angular_angular"
# repo_path="2_7/apache_spark"
# repo_path="2_7/apache_kafka"
repo_path="data/2_7/facebook_react"
# repo_path="2_8/django_django"
# repo_path="smalldata/ftr"
# repo_path="2_7/julialang_julia"
# repo_path="2_7/ruby_ruby"
# repo_path="2_8/pytorch_pytorch"
# repo_path="2_9/huggingface_transformers"
# repo_path="2_9/redis_redis"

# code_df_cache="data/merged_code_df/multi_code_df.parquet"
code_df_cache="data/2_7/facebook_react/cache/repr_0.1663/code_df.parquet"


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
train_commits=1500 # number of commits to train on (train + val)
psg_cnt=25 # number of commits to use for psg generation
psg_len=350
psg_stride=250
aggregation_strategy="sump" # aggregation strategy for bert reranker
rerank_depth=100 # depth to go while reranking
openai_model="gpt4" # openai model to use
# bert_best_model="${repo_path}/models/microsoft_codebert-base_bertrr_gpt_train/best_model"
# bert_best_model="2_7/facebook_react/models/microsoft_codebert-base_bertrr_gpt_train/best_model"
bert_best_model="data/combined_commit_train/best_model"

best_model_path="data/2_7/facebook_react/models/bce/best_model"

# repo_paths=(
#     "data/2_7/apache_spark"
#     "data/2_7/apache_kafka"
#     "data/2_7/facebook_react"
#     "data/2_8/angular_angular"
#     "data/2_8/django_django"
#     "data/2_8/pytorch_pytorch"
#     "data/2_7/pandas-dev_pandas"
#     "data/2_7/julialang_julia"
#     "data/2_7/ruby_ruby"
#     "data/2_8/ansible_ansible"
#     "data/2_7/moby_moby"
#     "data/2_7/jupyter_notebook"
# )

python -u src/bce.py \
    --repo_path $repo_path \
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
    --psg_len $psg_len \
    --psg_stride $psg_stride \
    --use_gpu \
    --aggregation_strategy $aggregation_strategy \
    --rerank_depth $rerank_depth \
    --openai_model $openai_model \
    --eval_folder $eval_folder \
    --eval_gold \
    --use_gpt_train \
    --triplet_mode $triplet_mode \
    --bert_best_model $bert_best_model \
    --code_df_cache $code_df_cache \
    --best_model_path $best_model_path \

    # --do_eval \
    # --do_train \
    # --use_previous_file \
    # --debug


    # --sanity_check \
    # --overwrite_cache \
    # --ignore_gold_in_training \
    # --do_combined \
    # --repo_paths "${repo_paths[@]}" \


echo "Job completed"









# repo_path="2_8/pytorch_pytorch" ???????
# repo_path="2_7/pandas-dev_pandas" ???????
# repo_path="2_7/julialang_julia"
# repo_path="2_7/ruby_ruby"
# repo_path="2_8/ansible_ansible"
# repo_path="2_7/moby_moby"
# repo_path="2_7/jupyter_notebook"