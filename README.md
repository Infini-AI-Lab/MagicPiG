# MagicPIG: LSH Sampling for Efficient LLM Generation

## installation

    bash install.sh

Only Intel CPUs are supported now. We also provide huggingface-like implementation for accuracy evaluation, which does not need Intel CPUs.

## Experiments

    cd RULER/RULER/scripts
    K=10 # LSH hyper-parameter for MagicPIG and Page Size for Quest
    L=150 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
    sink=4 # sink token
    local=64 # local token
    model=0 # 0: MagicPIG; 1: Quest; 2: TopK 3: Oracle Sampling
    expid=0
    bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid

This script is implemented in huggingface to reproduce accuracy results (for RULER benchmark). The reference file (for model and KV cache implementation) can be found in refs/.

Three models are supported now: llama3-8b-chat-128k (Llama-3.1-8B-Instruct), llama3-70b-chat-128k (Llama-3.1-70B-Instruct), mistral-7b-chat-512k (
MegaBeam-Mistral-7B-512k).

In models/, we implement MagicPIG CPU/GPU codes for sanity check and benchmarking.  models/magicpig_llama.py and models/cache.py are expected to be equivalent to refs/hf_model_ref.py and refs/hf_cache_ref.py. 

To benchmark the speed of MagicPIG

    cd models
    OMP_NUM_THREADS=96 python benchmark.py --P 98000 --M 98304 --B 1 --model meta-llama/Meta-Llama-3.1-8B-Instruct

To achieve best performance, currently you need to manually set the omp threads in lsh/lsh.cc and attention/gather_gemv.cc (as well as here) to match the number of physical cores in your CPUs.

For generation purpose,  

    cd models
    python generation.py --path ../data/data32k.json

where path specifies the input contexts.

models/magicpig_config.json is used to adjust proper hyper-parameters such as (K,L) in LSH algorithms and which layer to keep in GPUs.
