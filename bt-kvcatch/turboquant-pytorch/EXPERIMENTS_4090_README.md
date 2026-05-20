# Experiments 4090 README

This document is the execution guide for the formal KVcatch experiments on a
server with two RTX 4090 GPUs.

## Goal

We want to compare:

- `FP16`
- `SKVQ Baseline`
- `TurboQuant Baseline`
- `Hybrid+SKVQ+Block`
- `Hybrid+TQ+Block`
- `Hybrid+TQ+Block+PageMix`

and report:

- different page quantization schemes
- different benchmarks
- compression ratio
- theoretical page bpw
- effective whole-cache bpw

## Final Tables

Main result table:

| Method | Group | Backend | Page Policy | Page Quant Scheme | Reorder | Importance Metric | K bpw | V bpw | Avg bpw | Effective bpw | NIAH@2K | NIAH@4K | NIAH@8K | PPL | NarrativeQA | Qasper | MultiFieldQA | Tok/s |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scheme ablation table:

| Scheme ID | Method | Page Quant Scheme | High Ratio | Reorder | Importance Metric | K bpw | V bpw | Avg bpw | Effective bpw | NIAH@8K | PPL | LongBench Avg |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

Resource table:

| Method | Avg bpw | Effective bpw | Compression Ratio | Peak Memory | Tok/s | NIAH@8K |
|---|---:|---:|---:|---:|---:|---:|

## Notes On BPW

Use two bpw values in the final report:

- `Avg bpw`: theoretical bit width over old compressed pages only
- `Effective bpw`: true whole-cache average, computed from compression ratio

Recommended formula:

```text
Effective bpw = 16 / compression_ratio
```

If you want to report K/V pair bpw instead, keep the whole report consistent:

```text
Effective bpw(KV pair) = 32 / compression_ratio
```

## Step 0

Prepare the repo:

```powershell
cd D:\KVcatch
git pull origin master
cd D:\KVcatch\bt-kvcatch\turboquant-pytorch
pip install -r requirements.txt
pip install -e .
python -m turboquant.block_cache.test_block_cache
```

## Step 1

Generate reorder metadata once:

```powershell
python -m turboquant.block_cache.calibration `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --inline-text "Calibration text for K and V projection statistics." `
  --n-samples 32 `
  --seq-len 1024 `
  --metric absmax_sort `
  --key-group-size 128 `
  --value-group-size 64 `
  --output runs\llama32_3b_calib\reorder_meta.pt
```

## GPU 0

Run the formal main experiment:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -m turboquant.block_cache.experiment_main `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --context-lengths 2048,4096,8192 `
  --positions 0.1,0.5,0.9 `
  --seeds 0,1,2 `
  --max-new-tokens 32 `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --importance-metric k_norm `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --max-cached-decompressed-blocks 128 `
  --include-random-mix `
  --output-dir runs\server_main_exp
```

Then run the memory and speed profile:

```powershell
python -m turboquant.block_cache.profile_memory `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend all `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --context-length 4096 `
  --position 0.5 `
  --seed 0 `
  --max-new-tokens 32 `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --importance-metric k_norm `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --max-cached-decompressed-blocks 128 `
  --output-dir runs\server_profile
```

## GPU 1

Run PPL:

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python -m turboquant.block_cache.eval_ppl `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend all `
  --dataset wikitext `
  --dataset-config wikitext-103-raw-v1 `
  --split test `
  --max-samples 64 `
  --max-tokens 8192 `
  --seq-len 1024 `
  --stride 512 `
  --policy hybrid `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --importance-metric k_norm `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --max-cached-decompressed-blocks 128 `
  --output runs\server_ppl\ppl.jsonl
```

Run LongBench:

```powershell
python -m turboquant.block_cache.eval_longbench `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend all `
  --subsets narrativeqa,qasper,multifieldqa_en `
  --max-samples 16 `
  --policy hybrid `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --importance-metric k_norm `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --max-cached-decompressed-blocks 128 `
  --output-dir runs\server_longbench
```

Run page-quantization scheme ablation:

```powershell
python -m turboquant.block_cache.ablation `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_tq_mix `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --context-lengths 2048,4096,8192 `
  --positions 0.1,0.5,0.9 `
  --seeds 0,1,2 `
  --max-new-tokens 32 `
  --policy hybrid `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --max-cached-decompressed-blocks 128 `
  --sweep important_ratio=0.2,0.3,0.5 `
  --sweep high_key_bits=4,6 `
  --sweep high_value_bits=4 `
  --sweep low_key_bits=2 `
  --sweep low_value_bits=2 `
  --sweep block_size=8,16,32 `
  --output-dir runs\server_ablation_scheme
```

Run importance-metric ablation:

```powershell
python -m turboquant.block_cache.ablation `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_tq_mix `
  --reorder-file runs\llama32_3b_calib\reorder_meta.pt `
  --context-lengths 2048,4096 `
  --positions 0.1,0.5,0.9 `
  --seeds 0,1,2 `
  --max-new-tokens 32 `
  --policy hybrid `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --key-bits 2 `
  --value-bits 2 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --protected-layers 1 `
  --protected-key-bits 8 `
  --protected-value-bits 8 `
  --key-group-size 128 `
  --value-group-size 64 `
  --max-cached-decompressed-blocks 128 `
  --sweep importance_metric=k_norm,kv_norm,vk_ratio,random `
  --output-dir runs\server_ablation_metric
```

## Output Directories

Expected output directories:

- `runs\llama32_3b_calib`
- `runs\server_main_exp`
- `runs\server_profile`
- `runs\server_ppl`
- `runs\server_longbench`
- `runs\server_ablation_scheme`
- `runs\server_ablation_metric`

## Suggested Reporting Order

1. Main table: `server_main_exp`
2. PPL table: `server_ppl`
3. LongBench table: `server_longbench`
4. Resource table: `server_profile`
5. Scheme ablation: `server_ablation_scheme`
6. Importance ablation: `server_ablation_metric`

## Recommended Default Interpretation

- Main result uses `k_norm` as the default importance metric.
- `attention_score` should be treated as an ablation, not the primary result.
- Reorder should be enabled for both SKVQ and TurboQuant in the formal runs.
