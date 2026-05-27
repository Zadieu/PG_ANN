# Bench Comparison

- Baseline: `baseline`
- Candidate: `candidate`
- Matched runs: 4

| approx_kind | beam_width | l_search | baseline_qps | candidate_qps | delta_qps | baseline_recall | candidate_recall | delta_recall | baseline_latency_ms | candidate_latency_ms | delta_latency_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 2 | 6 | 702.420494 | 702.420494 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.423649 | 1.423649 | 0.000000 |
| full | 3 | 6 | 717.698713 | 717.698713 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.393342 | 1.393342 | 0.000000 |
| pq | 2 | 6 | 360.072648 | 360.072648 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 2.777217 | 2.777217 | 0.000000 |
| pq | 3 | 6 | 350.195397 | 350.195397 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 2.855549 | 2.855549 | 0.000000 |
