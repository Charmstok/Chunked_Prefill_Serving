# Experiments

This directory stores reproducible experiment scripts built around scheduling strategies. At the moment it mainly contains two classes of experiments:

- `heterogeneous_prompt/`
  - Used to validate the difference between fixed `chunk_size` scheduling and fixed `target_time` scheduling under strongly heterogeneous prompt workloads.
- `active_prefill_control/`
  - Used to validate whether control strategies for the activity level of unfinished prefills, such as `Active Prefill Control (APC)`, can change batch structure and, in turn, affect `prefill_e2e_time` and `request_e2e_time`.

All experiment scripts are expected to run from the project root and write results to timestamped directories under `./offline_inference_output/`.

---

## 1. Directory overview

The current experiment directory structure is:

- `example/experiment/README.md`
  - This document.
- `example/experiment/heterogeneous_prompt/`
  - Experiments for heterogeneous prompt workloads.
- `example/experiment/active_prefill_control/`
  - On/off comparison experiments for APC.

If you add more experiments later, each experiment subdirectory should ideally contain at least:

- one shared parameter file or utility file
- one or more directly runnable experiment scripts
- a local README describing experiment-specific parameters and outputs

---

## 2. Common execution conventions

### 2.1 Working directory

All scripts are expected to be run from the project root:

```bash
cd /home/ta/project/sarathi-serve
```

### 2.2 Python environment

Use the project virtual environment consistently:

```bash
./env/bin/python ...
```

### 2.3 Output location

Each run creates a separate directory under `./offline_inference_output/`. The directory name typically includes:

- a timestamp
- the experiment name
- an experiment label such as `on/off`

Typical outputs include:

- `config.json`
  - Stores the key configuration, script path, and workload summary for the run.
- `replica_0/heterogeneous_prompts.json`
  - Stores the heterogeneous prompt dataset actually used by the run.
- `replica_0/select_stats_rank0.csv`
  - Stores per-batch scheduling selection statistics.
- `replica_0/sequence_metrics.csv`
  - Stores request-level metrics such as `prefill_e2e_time` and `request_e2e_time`.

If an experiment writes additional statistics files, they are documented either in that experiment's own README or in the relevant section below.

### 2.4 Reproducibility

The current experiment scripts generally use the following reproducibility design:

- fixed data source: `dataset/ShareGPT_V3_unfiltered_cleaned_split.json`
- fixed seed for heterogeneous prompt construction
- fixed prompt length ranges and short/long ratio
- fixed arrival pattern, such as clustered arrival

As long as the model, tokenizer, and data source stay the same, the same script parameters should usually reproduce the workload.

---

## 3. `heterogeneous_prompt`

### 3.1 Experiment goal

This experiment constructs highly heterogeneous prompt workloads to test whether time-budget scheduling performs better than fixed `chunk_size` scheduling when the `prefill/decode` mix changes significantly.

The key questions are:

- whether the prompt length distribution is heterogeneous enough
- whether the `prefill/decode` ratio fluctuates significantly
- whether fixed `target_time` adapts better than fixed `chunk_size` under such heterogeneous workloads

### 3.2 Script description

There are currently two scripts:

- `example/experiment/heterogeneous_prompt/chunk_size.py`
  - Uses `SarathiSchedulerConfig`
  - Fixes `chunk_size` and observes batch formation behavior under heterogeneous prompt workloads
- `example/experiment/heterogeneous_prompt/target_time.py`
  - Uses `OptSarathiSchedulerConfig`
  - Fixes `target_time` and observes the benefit of time-budget scheduling under heterogeneous workloads

### 3.3 Workload construction

Neither script reads a normal prompt list sequentially anymore. Instead, both use
`sarathi.utils.prompt_utils.build_heterogeneous_prompt_dataset(...)`
to construct a two-segment, extremely heterogeneous prompt workload:

- extract the first human prompt from `dataset/ShareGPT_V3_unfiltered_cleaned_split.json`
- keep only two kinds of prompts:
  - short prompts: `10 ~ 30` tokens
  - long prompts: `500 ~ 512` tokens
- the `short / long` ratio is configurable and defaults to `1:1`
- request order is shuffled after satisfying the short/long ratio, but remains reproducible with the same `seed`
- assign `num_decode_tokens` separately for each prompt
- short prompts default to the `decode_heavy` regime
- long prompts default to the `prefill_heavy` regime
- each request is submitted to the engine with its own `SamplingParams(max_tokens=record["num_decode_tokens"])`
- the scripts set `SARATHI_SAMPLING_BACKEND=torch` by default to avoid unstable `flashinfer` sampling warnings under highly heterogeneous workloads

As a result, this workload simultaneously provides:

- large variation in prefill token counts
- large variation in decode token counts
- clear changes in the `prefill/decode` ratio
- reproducibility under the same parameter set and `seed`

### 3.4 Default experiment parameters

Both scripts currently use the same heterogeneous prompt parameters:

- `PROMPTS_NUMBER = 200`
- `model_config.max_model_len = 1024`
- `HETEROGENEOUS_PROMPT_SEED = 42`
- `SHORT_PROMPT_MIN_TOKENS = 10`
- `SHORT_PROMPT_MAX_TOKENS = 30`
- `LONG_PROMPT_MIN_TOKENS = 500`
- `LONG_PROMPT_MAX_TOKENS = 512`
- `SHORT_LONG_RATIO = (1, 1)`
- `MIN_DECODE_TOKENS = 16`

### 3.5 Output files

Each run additionally saves the following under its `output_dir`:

- `config.json`
  - Includes scheduling parameters, heterogeneous prompt parameters, and the workload summary for the run
- `replica_0/heterogeneous_prompts.json`
  - Includes the prompt text, `prompt_type`, `num_prefill_tokens`, `num_decode_tokens`, `pd_ratio`, and `decode_regime` actually used in the experiment

### 3.6 How to run

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/heterogeneous_prompt/chunk_size.py
./env/bin/python example/experiment/heterogeneous_prompt/target_time.py
```

### 3.7 Tuning suggestions

If you want to make the workload even more heterogeneous, prioritize tuning:

- the gap between the short and long prompt ranges
- `SHORT_LONG_RATIO`
- the lower bound for long prompts
- `model_config.max_model_len`

If you want the experiment to remain strictly reproducible, try not to change:

- `DATA_SOURCE`
- `HETEROGENEOUS_PROMPT_SEED`
- the tokenizer-aligned model

---

## 4. `active_prefill_control`

### 4.1 Experiment goal

This experiment compares behavior with `Active Prefill Control (APC)` disabled versus enabled under the same workload. Here, APC refers to controlling the activity level of unfinished prefills, rather than statically reserving fixed slots.

The experiment mainly answers three questions:

- after enabling APC, are unfinished prefills actually scheduled in a controlled active manner, rather than never being triggered at all
- after enabling APC, does batch structure change, for example with larger average prefill chunks or fewer decode-only batches
- do those structural changes ultimately affect `prefill_e2e_time` and `request_e2e_time`

### 4.2 Script description

This directory currently contains:

- `example/experiment/active_prefill_control/common.py`
  - shared parameters, workload construction, and the experiment entry point
- `example/experiment/active_prefill_control/baseline.py`
  - APC disabled, used as the control group
- `example/experiment/active_prefill_control/enabled.py`
  - APC enabled, used as the treatment group
- `example/experiment/active_prefill_control/run_pair.py`
  - runs `baseline.py` and `enabled.py` in sequence, then automatically calls the comparison script
- `example/experiment/active_prefill_control/compare_runs.py`
  - aggregates results from two runs and reports means, P90 values, and APC behavior statistics
- `example/experiment/active_prefill_control/README.md`
  - the short README for this experiment directory

### 4.3 Current workload characteristics

`common.py` currently constructs a clustered workload that is more `decode-heavy`, in order to amplify the competition between unfinished prefills and decodes. Key parameters include:

- `PROMPTS_NUMBER = 240`
- `TARGET_TIME = 100`
- `ARRIVAL_INTERVAL_S = 0.025`
- `CLUSTER_START_PCT = 0.10`
- `CLUSTER_END_PCT = 0.30`
- `SHORT_PROMPT_MIN_TOKENS = 30`
- `SHORT_PROMPT_MAX_TOKENS = 50`
- `LONG_PROMPT_MIN_TOKENS = 200`
- `LONG_PROMPT_MAX_TOKENS = 220`
- `SHORT_LONG_RATIO = (49, 1)`
- `MIN_DECODE_TOKENS = 4`
- `MAX_MODEL_LEN = 256`
- `CHUNK_SIZE = 512`
- `MAX_NUM_SEQS = 32`
- `GPU_MEMORY_UTILIZATION = 0.65`

The default APC-related parameters are:

- `MAX_ACTIVE_PREFILL_SEQS = 6`
- `MIN_ACTIVE_PREFILL_CHUNK_SIZE = 16`

### 4.4 Output files

In addition to the common output files, this experiment also generates:

- `replica_0/active_prefill_control_stats_rank0.csv`
  - the per-batch APC behavior statistics file
  - key fields include:
    - `active_prefill_seq_cap`
    - `active_prefill_seq_count`
    - `deferred_prefill_seq_count`
    - `waiting_prefill_blocked_by_cap`
    - `waiting_prefill_blocked_by_min_chunk`
    - `scheduled_prefill_seq_count`
    - `scheduled_prefill_tokens`
    - `avg_prefill_chunk`

`compare_runs.py` currently reads:

- `replica_0/sequence_metrics.csv`
- `replica_0/active_prefill_control_stats_rank0.csv`

and reports:

- `request_e2e_mean`
- `prefill_e2e_mean`
- `request_e2e_p90`
- `prefill_e2e_p90`
- `scheduled_prefill_seq_mean`
- `avg_prefill_chunk_mean`
- `deferred_prefill_seq_total`
- `blocked_by_cap_total`
- `blocked_by_min_chunk_total`
- `delta:on-off`

### 4.5 How to run

Run individually:

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/baseline.py
./env/bin/python example/experiment/active_prefill_control/enabled.py
```

Use the automatic comparison script:

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/run_pair.py
```

If you already have an `off/on` pair of output directories, you can also pass them to the comparison script manually:

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/compare_runs.py \
  /path/to/off_dir \
  /path/to/on_dir
```

### 4.6 Suggested metrics to inspect

When analyzing APC experiments, start with the following groups of metrics:

- request-side:
  - `prefill_e2e_time`
  - `request_e2e_time`
- batch-structure side:
  - `scheduled_prefill_seq_count`
  - `avg_prefill_chunk`
  - the proportion of decode-only batches
- APC behavior side:
  - `active_prefill_seq_count`
  - `deferred_prefill_seq_count`
  - `waiting_prefill_blocked_by_cap`
  - `waiting_prefill_blocked_by_min_chunk`

If `active_prefill_control_stats_rank0.csv` rarely shows `active_prefill_seq_count > 0` over time, it usually means one of the following:

- the current workload is still not easy enough to trigger APC
- `C_max` is too large, so APC degenerates into nearly unconstrained admission
- `L_min` is too small, so APC lacks a minimum useful-progress constraint
- or conversely, `L_min` is too large, making it difficult for prefills to enter the active set

---

## 5. Recommended order of use

If you are using these experiment scripts for the first time, the recommended order is:

1. Run `heterogeneous_prompt/target_time.py` first to confirm that the time-budget experiment path works correctly.
2. Then run `active_prefill_control/baseline.py` and `enabled.py` to confirm that the APC switch experiment produces `sequence_metrics.csv` and `active_prefill_control_stats_rank0.csv` correctly.
3. Finally, use `run_pair.py` or `compare_runs.py` for paired comparison.

---

## 6. Suggestions for future extensions

If you add more experiments under `example/experiment/` later, it is recommended to follow these conventions:

- each experiment subdirectory should contain at least one local README
- keep shared parameters in `common.py` or an equivalent file
- use `config.json` consistently to save the run configuration and workload summary
- if you add new scheduler behavior statistics, do not change the meaning of the existing `select_stats_rank0.csv`; prefer adding a separate statistics file
- if you provide `on/off` comparisons, also provide an automatic summarization script
