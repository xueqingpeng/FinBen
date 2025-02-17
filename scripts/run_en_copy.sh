#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

export HF_DATASETS_CACHE="/vast/palmer/scratch/xu_hua/xp83/saved_datasets"
export HF_MODELS_CACHE="/vast/palmer/scratch/xu_hua/xp83/saved_models"
export HF_HOME="/vast/palmer/scratch/xu_hua/xp83/saved_models"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1

# Array of models
MODELS=(
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "google/gemma-2-2b"
    # "internlm/internlm2_5-1_8b"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # "ibm-granite/granite-3.0-2b-base"
    # "nvidia/Llama-3.1-Minitron-4B-Depth-Base"
    # "nvidia/Llama-3.1-Minitron-4B-Width-Base"
    # "nvidia/Minitron-4B-Base"
    # "HuggingFaceTB/SmolLM2-135M"
    # "HuggingFaceTB/SmolLM2-360M"
    # "HuggingFaceTB/SmolLM2-1.7B"
    # "EleutherAI/pythia-410m"
    # "TheFinAI/OpenFinllm-ZH-0.5B"
    "TheFinAI/OpenFinllm-ZH-1.5B"
    # "TheFinAI/OpenFinllm-ZH-3B"
)

# Revisions for specific models
declare -A REVISIONS_MAP
# Array of revisions
REVISIONS_MAP["TheFinAI/OpenFinllm-ZH-0.5B"]="aa0ffe41a49ce922af97aa37085fbace6edb70b6 174a919a6e1e1a387ff33173fdaa6831e6349d78 2176c44d2866f33a7994ea775e4f761dd31db0f7 82af1f8824eaf80741e0a1f77daa5bd77e95e800 a7f852af77f9ad7ae40fc658f2cfc50c481d4f31 66f60df334d18dad7cd01ad858f61326223257af 10a6f96d316bc7a6facacf60c4a92ab258460159 85280e0c4df0c2fc7d0db780633eb0f575e54273"
REVISIONS_MAP["TheFinAI/OpenFinllm-ZH-1.5B"]="0c6a9308b67542dac18561b0fe25e199123af577" # "496d97906df88dc716801a0c3f9f4fe4b4e43a52 22ef85885aed921c9186282aae63d7474a59253e 9164052f71b051e90715cc76f162acc99fadc5a0 94f7e203f60113fefecf02b51de6ce426093cbef 409ad0e55d2f25f8a737b485ebff23cc70e07f0c 5ecac8d7938f7c0a7a153646d420d2c4180cc0bf 4e45319c45816b2ab82a2c29b95e7238dc5eee37 cc90133d54070d03fab20f0cd3f1873c99f2f938 c9b720793aa11cf4811c75860caf6d879b8ba69b 0c6a9308b67542dac18561b0fe25e199123af577"
REVISIONS_MAP["TheFinAI/OpenFinllm-ZH-3B"]="ae14269dc435e4afbc70e2c2ab627f481b3213e5 15984db3e9aeb84eb5a9e900bc5e5a2f0a1a7852 1b054b96b4a405070d05c39d4110d8486c10a30d eb52b73de022f266bf9c3e59e0c51b725319f310 ca7016aa7704b775010a8bc4f2bd86ebc0e5f14b 227a6f6c86853abc8044bde8a41856020c19158b 9824cb29d8ab88f26e64e9607086ed3a62a54e42 a6a5dd31be0fde954345c358ef58554742d2d863 32205f23e34ee4cf225b0936ce47de8f3147fbee df55b89d3f5521286c3ba49db2677608f9a835f2"

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "* Evaluating model: $MODEL"

    # Check if the model has multiple revisions
    if [[ -n "${REVISIONS_MAP[$MODEL]}" ]]; then
        # Loop through each revision
        for REVISION in ${REVISIONS_MAP[$MODEL]}; do
            echo "* Using revision: $REVISION"

            lm_eval --model vllm \
                --model_args "pretrained=$MODEL,revision=$REVISION,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024,trust_remote_code=True" \
                --tasks zh-reg-classification \
                --num_fewshot 5 \
                --batch_size auto \
                --output_path ../results \
                --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-zh-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
                --log_samples \
                --include_path ../../repo_from_others/zh_fin_reg/FinBen/tasks/chinese
            
            echo "* Finished evaluating revision: $REVISION"
        done
    else
        # Run models that don’t need a revision
        lm_eval --model vllm \
            --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024,trust_remote_code=True" \
            --tasks reg-qa \
            --num_fewshot 5 \
            --batch_size auto \
            --output_path ../results \
            --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-zh-5shot,push_results_to_hub=True,push_samples_to_hub=False,public_repo=False" \
            --log_samples \
            --include_path ../../repo_from_others/zh_fin_reg/FinBen/tasks/chinese
    fi

    echo "* Finished evaluating model: $MODEL"
done