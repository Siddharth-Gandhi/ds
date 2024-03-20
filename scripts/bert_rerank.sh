#!/bin/bash
#SBATCH --job-name=bp
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


eval_folder="bm25_fix_combined_bert_classification"
notes="test_out"


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

repo_name=$(echo $data_path | rev | cut -d'/' -f1 | rev)

train_mode="classification"

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
k=10000 # initial ranker depth
n=100 # number of samples to evaluate on

model_path="microsoft/codebert-base"
# model_path="microsoft/graphcodebert-base"


# overwrite_cache=False # whether to overwrite the cache
batch_size=32 # batch size for inference
num_epochs=5 # number of epochs to train
learning_rate=1e-5 # learning rate for training
num_positives=10 # number of positive samples per query
num_negatives=10 # number of negative samples per querys
train_depth=10000 # depth to go while generating training data
num_workers=8 # number of workers for dataloader
train_commits=2000 # number of commits to train on (train + val)
psg_cnt=5 # number of commits to use for psg generation
aggregation_strategy="sump" # aggregation strategy for bert reranker
# use_gpu=True # whether to use gpu or not
rerank_depth=250 # depth to go while reranking
output_length=1000 # length of the output in .teIn file
# do_train=True # whether to train or not
# do_eval=True # whether to evaluate or not
openai_model="gpt4" # openai model to use
# best_model_path="data/combined_commit_train/best_model"
# best_model_path="/home/ssg2/ssg2/ds/data/combined_gpt_train/combined_bce_train/best_model"

triplet_cache_path="/home/ssg2/ssg2/ds/cache/facebook_react/bert_reranker/combined_triplet_data.pkl"
# triplet_cache_path="/home/ssg2/ssg2/ds/cache/facebook_react/bert_reranker/bert_bce_fb/triplet_data_cache.pkl"
# best_model_path="data/combined_commit_train/best_model"
# best_model_path="models/facebook_react/bert_reranker/4xbert_bce_fb/best_model"

python -u src/BERTReranker.py \
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
    --use_gpu \
    --aggregation_strategy $aggregation_strategy \
    --rerank_depth $rerank_depth \
    --openai_model $openai_model \
    --eval_folder $eval_folder \
    --eval_gold \
    --use_gpt_train \
    --do_eval \
    --do_train \
    --train_mode $train_mode \
    --triplet_cache_path $triplet_cache_path \
    # --best_model_path $best_model_path \


    # --overwrite_eval \
    # --do_combined_train \
    # --repo_paths "${repo_paths[@]}" \
    # --sanity_check \
    # --overwrite_cache \
    # --ignore_gold_in_training \


    # --debug \
find models/$repo_name/"bert_rerank"/$eval_folder -type d -name 'checkpoint*' -exec rm -rf {} +
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