# active_prefill_control

This experiment compares two configurations:

- `baseline.py`: disables active prefill control.
- `enabled.py`: enables a concurrency cap on active prefills and a minimum effective chunk-size constraint.

Shared workload characteristics:

- `max_num_seqs=32`
- `chunk_size=512`
- `target_time=100`
- clustered arrival
- short prompts dominate, and the short prompts correspond to a decode-heavy workload

Additional output file:

- `active_prefill_control_stats_rank0.csv`
  - `active_prefill_seq_cap`
  - `active_prefill_seq_count`
  - `deferred_prefill_seq_count`
  - `waiting_prefill_blocked_by_cap`
  - `waiting_prefill_blocked_by_min_chunk`
  - `scheduled_prefill_seq_count`
  - `scheduled_prefill_tokens`
  - `avg_prefill_chunk`

How to run:

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/baseline.py
./env/bin/python example/experiment/active_prefill_control/enabled.py
./env/bin/python example/experiment/active_prefill_control/compare_runs.py
```

Or run everything sequentially:

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/run_pair.py
```
