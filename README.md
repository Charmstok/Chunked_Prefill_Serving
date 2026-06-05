# Fairness-Aware and Latency-Controllable Scheduling for Chunked-Prefill LLM Serving

This project is a high-throughput, low-latency LLM inference framework. The corresponding paper is titled "Fairness-Aware and Latency-Controllable Scheduling for Chunked-Prefill LLM Serving."

It builds on [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) with further optimizations.

---

## Current Work

### Supported models

- Qwen3/Qwen3-8B
- Qwen/Qwen-7B
- openPangu(https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1)

### Tested dataset

- [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main)

---

## Setup

### CUDA

This project was developed and tested with CUDA 12.8 on NVIDIA 3090 and 4090 GPUs.
Some experiments were also run on NVIDIA RTX PRO 5000 72GB GPUs.

### Clone the project

```sh
git clone https://github.com/Charmstok/sarathi-serve.git
```

### Conda environment

Create a Python 3.11 environment:

```sh
conda create -p ./env python=3.11
```

Activate the environment:

```sh
conda activate ./env
```

### Install the project dependencies and CUDA-related requirements

```sh
pip install -e .
pip install -r requirements.txt
```

### How do I use this project?

#### Option 1: Basic chat

```shell
python example/chat/chat_only.py
```

#### Option 2: Offline evaluation

You can find the benchmark metrics in the `offline_inference_output` directory.

```shell
python example/offline_inference/target_time.py
```

Before using `target_time`, first obtain the time prediction model as described below.

Run the offline data collection script:

```shell
python example/time_balance/offline_select_status_csv.py
```

It runs a batch of requests with `SarathiScheduler` and writes `select_stats_rank0.csv` during model execution:

> Notes:
> - The script output directory is `offline_inference_output/<timestamp>/replica_0/`.
> - The key artifact is `select_stats_rank0.csv`, which contains features and labels such as `decode_tokens/prefill_tokens/.../latency_ms`.
> - If you changed parameters such as `chunk_size/max_num_seqs/max_model_len` in the script, keep the training configuration consistent, especially `chunk_size`.

Next, open `sarathi/time_balance/config.py` and update it based on the data you just generated:
- `CSV_PATH`: point this to the latest `select_stats_rank0.csv`
- `MODEL_CACHE_PATH`: path where the model will be saved (default: `sarathi/time_balance/time_predictor_mlp.pt`)
- `BUCKET_SPLIT_CONFIG.chunk_size`: recommended to match the scheduler `chunk_size` used during data collection

Run:

```sh
python sarathi/time_balance/predict_time.py
```

After training, the script prints the train/val/test MAE and writes the model to `MODEL_CACHE_PATH`.

`OptSarathiScheduler` loads the model from `MODEL_CACHE_PATH` in `sarathi/time_balance/config.py` during initialization. If the file does not exist, it raises an `assert` immediately to avoid silent degradation in production.

Use the following script to quickly verify that the model can be loaded and used for prediction:

```sh
python sarathi/time_balance/load_model.py
```

The commands above do not cover every parameter. For more options, run:

```sh
python -m sarathi.entrypoints.api_server --help
```

### Tuning scripts

If you want higher performance, you may need to tune some parameters.

See [Parameter Tuning README.md](example/README.md) for details.

---

## Acknowledgements

This repository originated from a branch of the [sarathi-serve](https://github.com/microsoft/sarathi-serve) project.
