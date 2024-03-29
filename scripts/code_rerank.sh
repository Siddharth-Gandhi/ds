#!/bin/bash
#SBATCH --job-name=cr_or_f2
#SBATCH --output=logs/code_rerank/output_%A.log
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=boston-2-35

# Activate the conda environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "On host $(hostname)"
nvidia-smi
export TOKENIZERS_PARALLELISM=true


eval_folder="eval_cr_oracle_final"
notes="test_out"
# triplet_mode="parse_functions"
# triplet_mode="sliding_window"
triplet_mode="diff_content"
# triplet_mode="diff_subsplit"



# data_path="2_8/angular_angular"
# data_path="2_7/apache_spark"
# data_path="2_7/apache_kafka"
data_path="data/2_7/facebook_react"
# data_path="2_8/django_django"
# data_path="smalldata/ftr"
# data_path="2_7/julialang_julia"
# data_path="2_7/ruby_ruby"
# data_path="2_8/pytorch_pytorch"
# data_path="2_9/huggingface_transformers"
# data_path="2_9/redis_redis"


# split data path on / and take the last element
repo_name=$(echo $data_path | rev | cut -d'/' -f1 | rev)

code_df_cache="cache/facebook_react/code_reranker/fb_code_df.parquet"
# code_df_cache="data/merged_code_df/multi_code_df.parquet"
# code_df_cache="data/2_7/facebook_react/cache/repr_0.1663/code_df.parquet"
# triplet_cache_path="data/2_7/facebook_react/cache/combined_diffs/diff_code_triplets.parquet"



index_path="${data_path}/index_commit_tokenized"
k=10000 # initial ranker depth
n=100 # number of samples to evaluate on
# no_bm25=False # whether to use bm25 or not

model_path="microsoft/codebert-base"
# model_path="microsoft/graphcodebert-base"



# overwrite_cache=False # whether to overwrite the cache
batch_size=32 # batch size for inference
num_epochs=5 # number of epochs to train
learning_rate=1e-5 # learning rate for training
num_positives=10 # number of positive samples per query
num_negatives=10 # number of negative samples per querys
output_length=1000
train_depth=10000 # depth to go while generating training data
num_workers=8 # number of workers for dataloader
train_commits=2000 # number of commits to train on (train + val)
psg_cnt=25 # number of commits to use for psg generation
psg_len=350
psg_stride=250
aggregation_strategy="maxp" # aggregation strategy for both bert reranker and codebert reranker
rerank_depth=100 # depth to go while reranking
openai_model="gpt4" # openai model to use


# BERT paths
# bert_best_model="${data_path}/models/microsoft_codebert-base_bertrr_gpt_train/best_model"
# bert_best_model="2_7/facebook_react/models/microsoft_codebert-base_bertrr_gpt_train/best_model"
# bert_best_model="data/combined_commit_train/best_model"
bert_best_model="/home/ssg2/ssg2/ds/models/facebook_react/bert_reranker/bm25_fix_combined_bert_classification/best_model"


# CodeReranker paths
# best_model_path="data/2_7/facebook_react/models/bce/best_model"
best_model_path="data/2_7/facebook_react/models/combined_diffs/best_model"
# best_model_path="data/2_7/facebook_react/models/X_function_split/best_model"
# best_model_path="data/2_7/facebook_react/models/combined_random/best_model"


# train_mode="regression"


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

python -u src/CodeReranker.py \
    --data_path $data_path \
    --index_path $index_path \
    --k $k \
    --n $n \
    --output_length $output_length \
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
    # --debug
    # --do_eval \
    # --filter_invalid \

    # --do_train \
    # --train_mode $train_mode \
    # --triplet_cache_path $triplet_cache_path \

    # --use_previous_file \
    # --debug




    # --sanity_check \
    # --overwrite_cache \
    # --ignore_gold_in_training \
    # --do_combined \
    # --repo_paths "${repo_paths[@]}" \

find models/$repo_name/"code_rerank"/$eval_folder -type d -name 'checkpoint*' -exec rm -rf {} +
echo "Job completed"









# data_path="2_8/pytorch_pytorch" ???????
# data_path="2_7/pandas-dev_pandas" ???????
# data_path="2_7/julialang_julia"
# data_path="2_7/ruby_ruby"
# data_path="2_8/ansible_ansible"
# data_path="2_7/moby_moby"
# data_path="2_7/jupyter_notebook"