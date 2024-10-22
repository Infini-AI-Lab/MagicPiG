# MagicPIG: LSH Sampling for Efficient LLM Generation

## installation

    bash install.sh

Only Intel CPUs are supported now. We also provide a huggingface-like implementation for accuracy evaluation, which does not need Intel CPUs.

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

In models/, we implement MagicPIG CPU/GPU codes for sanity checks and benchmarking.  models/magicpig_llama.py and models/cache.py are expected to be equivalent to refs/hf_model_ref.py and refs/hf_cache_ref.py. 

To benchmark the speed of MagicPIG

    cd models
    OMP_NUM_THREADS=96 python benchmark.py --P 98000 --M 98304 --B 1 --model meta-llama/Meta-Llama-3.1-8B-Instruct

To achieve the best performance, currently you need to manually set the omp threads in lsh/lsh.cc and attention/gather_gemv.cc (as well as here) to match the number of physical cores in your CPUs.

For generation purposes,  

    cd models
    python generation.py --path ../data/data32k.json

where path specifies the input contexts.

models/magicpig_config.json adjusts proper hyper-parameters such as (K, L) in LSH algorithms and which layer to keep in GPUs.

```bibtex
@misc{chen2024magicpiglshsamplingefficient,
      title={MagicPIG: LSH Sampling for Efficient LLM Generation}, 
      author={Zhuoming Chen and Ranajoy Sadhukhan and Zihao Ye and Yang Zhou and Jianyu Zhang and Niklas Nolte and Yuandong Tian and Matthijs Douze and Leon Bottou and Zhihao Jia and Beidi Chen},
      year={2024},
      eprint={2410.16179},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16179}, 
}
```
