# SIFT1M Cache Comparison

Run time: 2026-05-31 21:27 CST

Dataset: SIFT1M

Common configuration:

- Search list `L`: `25 30 35 40 45`
- Beam width: `8`
- Threads: `8`
- `DECO_IMPL=1`
- `USE_DISK_GRAPH_CACHE_INDEX=0`
- `MEM_GRAPH_USE_RATIO=0.1`
- `MEM_EMB_USE_RATIO=0.0`
- `CACHE=0`
- `K=10`

Compared cache schemes:

- Static direct fill: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO` unset, so all 100000 graph-cache slots are loaded statically.
- Static direct fill rerun: same as static direct fill, rerun at 2026-05-31 21:51 CST.
- 1% dynamic: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.01`, so 99000 graph-cache slots are static and 1000 slots are dynamic.
- 2% dynamic: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.02`, so 98000 graph-cache slots are static and 2000 slots are dynamic.
- 3% dynamic: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.03`, so 97000 graph-cache slots are static and 3000 slots are dynamic.
- 5% dynamic: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.05`, so 95000 graph-cache slots are static and 5000 slots are dynamic.
- 10% dynamic: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.1`, so 90000 graph-cache slots are static and 10000 slots are dynamic.

## Latest Sweep Results

This table uses the rerun static baseline and the latest 1%, 2%, and 3% dynamic runs.

| L | Static Rerun QPS | Dynamic 1% QPS | Dynamic 2% QPS | Dynamic 3% QPS | Static Rerun Recall@10 | Dynamic 1% Recall@10 | Dynamic 2% Recall@10 | Dynamic 3% Recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 2734.06 | 3650.67 | 3308.25 | 3684.04 | 90.70 | 90.66 | 90.71 | 90.70 |
| 30 | 2803.31 | 3596.76 | 3151.35 | 3377.26 | 92.83 | 92.98 | 92.98 | 92.96 |
| 35 | 2766.02 | 3341.54 | 3036.97 | 3179.58 | 94.53 | 94.61 | 94.54 | 94.54 |
| 40 | 2606.99 | 3073.25 | 2863.39 | 3034.57 | 95.64 | 95.68 | 95.74 | 95.66 |
| 45 | 2442.58 | 2887.80 | 2757.64 | 2857.35 | 96.50 | 96.51 | 96.53 | 96.55 |

## Latest QPS Change vs Static Rerun

| L | Dynamic 1% | Dynamic 2% | Dynamic 3% |
|---:|---:|---:|---:|
| 25 | +33.53% | +21.00% | +34.75% |
| 30 | +28.30% | +12.42% | +20.48% |
| 35 | +20.81% | +9.79% | +14.95% |
| 40 | +17.88% | +9.83% | +16.40% |
| 45 | +18.22% | +12.90% | +16.98% |

## Full Results

| L | Static QPS | Dynamic 3% QPS | Dynamic 5% QPS | Dynamic 10% QPS | Static Recall@10 | Dynamic 3% Recall@10 | Dynamic 5% Recall@10 | Dynamic 10% Recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 3320.30 | 3684.04 | 3354.21 | 2878.44 | 90.63 | 90.70 | 90.70 | 90.65 |
| 30 | 3365.32 | 3377.26 | 2421.45 | 1674.27 | 92.88 | 92.96 | 92.95 | 92.65 |
| 35 | 3294.05 | 3179.58 | 2496.08 | 1657.33 | 94.48 | 94.54 | 94.51 | 94.20 |
| 40 | 3074.80 | 3034.57 | 2536.78 | 1591.98 | 95.64 | 95.66 | 95.63 | 95.45 |
| 45 | 2898.80 | 2857.35 | 2476.15 | 1543.61 | 96.51 | 96.55 | 96.47 | 96.33 |

## QPS Change vs Static

| L | Dynamic 3% | Dynamic 5% | Dynamic 10% |
|---:|---:|---:|---:|
| 25 | +10.96% | +1.02% | -13.31% |
| 30 | +0.35% | -28.05% | -50.25% |
| 35 | -3.48% | -24.22% | -49.69% |
| 40 | -1.31% | -17.50% | -48.23% |
| 45 | -1.43% | -14.58% | -46.75% |

## Raw Static Direct Fill Result

```text
   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  3320.30     2403     4794    44.25     0.82    53.75  2883.44     0.08     8.00  1948.48     0.00    18.63   221.23   142.49      206     90.63
  30   8  3365.32     2374     4329    46.86     0.97    57.91  3032.13     0.03     7.00  1894.52     0.00    17.83   235.12   151.71      207     92.88
  35   8  3294.05     2424     4244    49.56     1.12    62.22  3183.62     0.02     6.93  1936.18     0.00    17.38   234.94   158.10      209     94.48
  40   8  3074.80     2597     5702    52.63     1.27    66.77  3347.03     0.06     8.54  2048.12     0.00    20.05   271.10   172.76      209     95.64
  45   8  2898.80     2756     6264    55.73     1.41    71.36  3513.62     0.05     9.86  2158.96     0.00    21.93   298.17   186.16      210     96.51
```

## Raw Static Direct Fill Rerun

```text
   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  2734.06     2920     5877    43.73     0.82    53.18  2851.27     0.06     6.35  2453.75     0.00    16.37   211.34   168.17      204     90.70
  30   8  2803.31     2850     4682    46.16     0.98    57.26  2994.15     0.09     7.29  2339.28     0.00    18.07   242.02   175.42      205     92.83
  35   8  2766.02     2889     4708    49.07     1.12    61.71  3154.33     0.06     7.75  2345.18     0.00    18.95   260.15   182.73      207     94.53
  40   8  2606.99     3065     4978    52.06     1.27    66.24  3317.04     0.05     8.63  2479.62     0.00    20.17   278.75   199.76      208     95.64
  45   8  2442.58     3271     5501    55.22     1.41    70.86  3484.25     0.07     8.92  2659.11     0.00    21.15   287.04   214.34      209     96.50
```

## Raw 1% Dynamic Result

```text
enabled dynamic graph cache. static slots: 99000, dynamic slots: 1000

   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  3650.67     2186     5091    31.23     0.82    36.95  2798.13     0.06    35.26  1645.47     0.00    57.25   230.81   155.90      217     90.66
  30   8  3596.76     2220     4513    33.79     0.97    40.58  2936.18     0.08    59.85  1564.25     0.00    59.54   308.16   164.39      217     92.98
  35   8  3341.54     2391     4786    36.78     1.12    44.71  3082.71     0.07    69.07  1658.89     0.00    61.45   353.68   179.34      230     94.61
  40   8  3073.25     2599     4393    39.92     1.27    48.97  3233.07     0.04    65.22  1850.27     0.00    63.08   349.28   198.81      230     95.68
  45   8  2887.80     2767     5230    43.10     1.41    53.26  3383.33     0.05    71.34  1964.46     0.00    64.15   379.21   211.20      230     96.51
```

## Raw 2% Dynamic Result

```text
enabled dynamic graph cache. static slots: 98000, dynamic slots: 2000

   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  3308.25     2412     6032    30.35     0.82    35.86  2797.45     0.04    40.61  1852.35     0.00    55.89   229.25   175.89      217     90.71
  30   8  3151.35     2535     4737    32.69     0.98    39.21  2933.75     0.02    65.32  1859.10     0.00    54.97   303.66   192.89      217     92.98
  35   8  3036.97     2630    11013    35.53     1.13    43.14  3071.02     0.04    75.56  1880.69     0.00    57.18   347.26   205.01      229     94.54
  40   8  2863.39     2790     4595    38.68     1.28    47.40  3220.89     0.06    89.02  1955.27     0.00    60.91   398.40   217.66      229     95.74
  45   8  2757.64     2897     5029    41.82     1.42    51.64  3367.92     0.06   112.46  1922.83     0.00    67.45   498.65   221.03      229     96.53
```

## Raw 3% Dynamic Result

```text
enabled dynamic graph cache. static slots: 97000, dynamic slots: 3000

   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  3684.04     2164     6675    29.72     0.83    35.16  2786.72     0.03    95.84  1402.88     0.00    60.04   390.68   154.16      217     90.70
  30   8  3377.26     2365     4683    31.87     1.00    38.30  2920.77     0.04   237.55  1081.14     0.00    62.70   763.48   157.34      217     92.96
  35   8  3179.58     2512     5188    34.83     1.15    42.33  3064.38     0.03   245.91  1153.23     0.00    67.45   808.50   169.85      230     94.54
  40   8  3034.57     2632     4689    37.89     1.30    46.50  3209.61     0.08   207.54  1339.65     0.00    66.97   756.98   190.08      230     95.66
  45   8  2857.35     2796     4951    41.11     1.45    50.78  3359.14     0.05   204.75  1475.32     0.00    68.23   767.53   204.51      230     96.55
```

## Raw 5% Dynamic Result

```text
enabled dynamic graph cache. static slots: 95000, dynamic slots: 5000

   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  3354.21     2378     6351    29.38     0.85    34.79  2789.55     0.03   199.00  1259.73     0.00    61.82   646.41   150.71      217     90.70
  30   8  2421.45     3301     6154    31.31     1.05    37.75  2928.47     0.05   357.24  1510.24     0.00    59.78  1099.69   213.28      217     92.95
  35   8  2496.08     3201     7337    34.09     1.22    41.65  3065.27     0.04   403.51  1256.00     0.00    60.14  1217.29   200.10      229     94.51
  40   8  2536.78     3148     5982    37.12     1.39    45.83  3208.46     0.02   463.91   997.33     0.00    64.96  1365.12   187.46      229     95.63
  45   8  2476.15     3226     5733    40.32     1.55    50.10  3357.43     0.02   352.82  1340.97     0.00    67.33  1176.88   216.52      229     96.47
```

## Raw 10% Dynamic Result

```text
enabled dynamic graph cache. static slots: 90000, dynamic slots: 10000

   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  25   8  2878.44     2776     7186    28.90     0.87    34.31  2789.68     0.04   517.38   856.55     0.00    55.43  1161.56   129.05      216     90.65
  30   8  1674.27     4770    11350    29.79     1.24    36.77  2908.32     0.01  1342.70   373.38     0.00    55.72  2815.78   125.88      216     92.65
  35   8  1657.33     4822    11061    32.53     1.49    40.86  3050.47     0.01  1277.34   423.77     0.00    61.51  2853.60   145.55      228     94.20
  40   8  1591.98     5020    10512    35.52     1.72    45.16  3195.34     0.01  1317.39   466.90     0.00    64.38  2946.26   160.82      228     95.45
  45   8  1543.61     5175    10027    38.68     1.92    49.54  3343.81     0.03  1209.82   721.86     0.00    66.50  2907.31   201.57      228     96.33
```

## Notes

- The latest rerun shows 1% dynamic as the strongest split among 1%, 2%, and 3% on this machine state.
- The static rerun is noticeably slower than the earlier static run, so repeated trials are needed before claiming a final speedup.
- In the earlier static-vs-dynamic sweep, 3% dynamic was the strongest dynamic split; in the latest rerun sweep, 1% dynamic is strongest.
- The 5% dynamic split is much better than the 10% dynamic split in this run.
- At `L=25` and `L=30`, 3% dynamic is slightly faster than static direct fill, with slightly higher recall.
- At `L=35` through `L=45`, 3% dynamic keeps recall slightly higher but QPS is within about 1-4% below static direct fill.
- At `L=25`, 5% dynamic is slightly faster than static direct fill while recall is also slightly higher.
- For `L=30` through `L=45`, 5% dynamic keeps recall nearly identical but is still slower than static direct fill.
- The 10% dynamic split keeps recall nearly unchanged, but throughput is much lower than static direct fill for this SIFT1M run.
- The slowdown appears mainly in dispatch and disk-processing time, which is consistent with the current dynamic cache path adding per-node bookkeeping and mutex-protected cache updates.
- These results test the non graph-replicated Gorgeous path because the current dynamic cache fill hook is in `page_search()`, not `page_search_dup_graph()`.

Log files:

- `/home/dell/data/gorgeous/sift1M/cache_compare/static_direct_cache_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/static_direct_cache_rerun_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/dynamic1_cache_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/dynamic2_cache_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/dynamic3_cache_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/dynamic5_cache_L25_45.log`
- `/home/dell/data/gorgeous/sift1M/cache_compare/dynamic10_cache_L25_45.log`
