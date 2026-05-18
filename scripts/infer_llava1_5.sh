#!/bin/bash

set -e

model_dir=llava-v1.5-7b-hf
model_path="model/$model_dir"

output_dir=outputs
max_new_tokens=512
device=cuda
tau=5
T=10

method=${1:-ilvad}
task=${2:-chair}

model_name=$model_dir

layers1=$(seq -s " " 0 30)
layers2=$(seq -s " " 5 25)

run_one() {
    dataset=$1
    seed=$2
    alpha=$3
    beta=$4
    layers=$5

    python main.py \
        --model_path "$model_path" \
        --device "$device" \
        --method "$method" \
        --alpha "$alpha" \
        --beta "$beta" \
        --tau "$tau" \
        --T "$T" \
        --layer_to_enhance $layers \
        --max_new_tokens "$max_new_tokens" \
        --seed "$seed" \
        --output_dir "$output_dir" \
        --datasets "$dataset"
}

if [[ "$task" == "pope" ]]; then

    alpha=5
    beta=1
    layers="$layers1"

    run_one pope_random 0 "$alpha" "$beta" "$layers"
    run_one pope_popular 0 "$alpha" "$beta" "$layers"
    run_one pope_adversarial 0 "$alpha" "$beta" "$layers"

    python eval/eval.py $output_dir/$model_name/pope_*_0-$method-$max_new_tokens.json --eval_pope

elif [[ "$task" == "chair" ]]; then

    alpha=5
    beta=0.3
    layers="$layers2"

    for seed in 0 1 2
    do
        run_one chair "$seed" "$alpha" "$beta" "$layers"
    done

    python eval/eval.py $output_dir/$model_name/chair_*-$method-$max_new_tokens.json --eval_chair

elif [[ "$task" == "llava_wild" ]]; then

    alpha=4
    beta=0.2
    layers="$layers2"

    run_one llava_wild 0 "$alpha" "$beta" "$layers"

else
    echo "Unknown task: $task"
    echo "Usage:"
    echo "  bash scripts/infer_llava1_5.sh ilvad pope"
    echo "  bash scripts/infer_llava1_5.sh ilvad chair"
    echo "  bash scripts/infer_llava1_5.sh ilvad llava_wild"
    exit 1
fi

wait
