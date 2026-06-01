# Gorgeous Experiment Results

Dataset: SIFT learn

Run time: 2026-05-31 17:47 CST

Configuration:

- Index: `M4_R64_L128`
- K: `10`
- Beam width: `8`
- Threads: `8`
- Cache nodes: `0`
- Page search: enabled
- Gorgeous graph-replicated mode: enabled

Raw result:

```text
   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  50   8  1417.13     5640    20459    42.65     4.81    47.46  2851.11     0.03     4.56  4741.46     0.00    15.94   165.97   639.05      171     98.35
 100   8  1079.76     7398    26974    66.79    12.90    79.69  3946.27     0.05    10.58  6287.27     0.00    30.22   247.21   703.15      171     99.81
```

Interpretation:

- `L=50`: faster, `QPS=1417.13`, `Recall@10=98.35`.
- `L=100`: more accurate, `Recall@10=99.81`, but slower at `QPS=1079.76`.
- `Mean Ltc` is average latency in microseconds.
- `999 Ltc` is 99.9 percentile latency in microseconds.

Original files:

- `/home/dell/data/gorgeous/sift_learn/M4_R64_L128/summary.log`
- `/home/dell/data/gorgeous/sift_learn/M4_R64_L128/search/search_K10_CACHE0_BW8_T8_MEML0_MEMK_PS1_USE_RATIO0.3_GP_LOCK_NUMS0_GP_CUT4096.log`
