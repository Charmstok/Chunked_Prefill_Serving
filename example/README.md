# Parameter Tuning

## Parameter 1: `prompt_weight` for the AGING policy

To choose better AGING parameters (`time_weight` and `prompt_weight`), the key is to put the two terms on the same scale and to be explicit about the objective you want to optimize: TTFT or overall E2E, throughput or fairness.

Priority formula: `priority = time_weight * wait_seconds + prompt_weight * remaining_prompt_tokens`

The simplest and most physically consistent approach is:
- Set `time_weight` to the number of prefill tokens that can be processed per second. You can derive this as `1 / seconds_per_prefill_token`, where `seconds_per_prefill_token` can be estimated from the benchmark results in `./offline_inference_output/{time}/replica_0/prefill_time_execution_plus_preemption_normalized.csv`, using either the P80 or the mean.
- Set `prompt_weight` to `-1.0`.

> If you want to prioritize throughput, reduce `time_weight` so the system favors shorter requests.
>
> For example, the default values `125 * 0.3` and `-1.0` bias the scheduler more toward short requests.

---

# Training the Time-Budget Model

The project's time-budget scheduler depends on the `TimePredictor` (an MLP regressor) trained by `sarathi/time_balance/predict_time.py`. The model is used during scheduling to predict the execution time of a batch in milliseconds.

The training workflow has two steps: first collect `select_stats_rank*.csv` offline, then train and save the model.

## 1. Collect training data offline (`select stats` CSV)

First run the offline script. It uses `SarathiScheduler` to process a batch of requests and writes `select_stats_rank0.csv` during model execution:

```sh
python example/time_balance/offline_select_status_csv.py
```

Notes:
- The script output directory is `offline_inference_output/<timestamp>/replica_0/`.
- The key artifact is `select_stats_rank0.csv`, which contains features and labels such as `decode_tokens/prefill_tokens/.../latency_ms`.
- If you changed parameters such as `chunk_size/max_num_seqs/max_model_len` in the script, keep the training configuration consistent, especially `chunk_size`.

## 2. Configure the training paths and bucketing parameters

Open `sarathi/time_balance/config.py` and update it based on the data you just generated:
- `CSV_PATH`: path to the latest `select_stats_rank0.csv`
- `MODEL_CACHE_PATH`: where the trained model will be saved (default: `sarathi/time_balance/time_predictor_mlp_v6.pt`)
- `BUCKET_SPLIT_CONFIG.chunk_size`: recommended to match the scheduler `chunk_size` used during data collection (default: `256`)

## 3. Train and save the model

Run:

```sh
python sarathi/time_balance/predict_time.py
```

After training, the script prints the train/val/test MAE and writes the model to `MODEL_CACHE_PATH`.

## 4. Usage notes (`OptSarathiScheduler`)

`OptSarathiScheduler` loads the model from `MODEL_CACHE_PATH` in `sarathi/time_balance/config.py` during initialization. If the file does not exist, it raises an `assert` immediately to avoid silent degradation in production.

Use the following script to quickly verify that the model loads correctly and can make predictions:

```sh
python sarathi/time_balance/load_model.py
```
