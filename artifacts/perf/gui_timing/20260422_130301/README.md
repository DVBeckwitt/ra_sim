# RA-SIM GUI Timing Report

- Scenario: `defaults`
- Trials: `6`
- Git: `main` `cb502991f964989c0a7515ec188649af426abe83`
- OS: `Windows-11-10.0.26200-SP0`
- Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- Backend: `TkAgg`
- State file: `none`
- Failed runs: `0`

## Startup

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
process launch to first visible | 6 | 0 | 6658.379 | 7389.221 | 6220.233 | 9748.664 | 9748.664 | 1480.161 | 8730.667, 9748.664, 6319.005, 6220.233, 6595.999, 6720.759
process launch overhead | 6 | 0 | 2.311 | 2.513 | 2.143 | 3.616 | 3.616 | 0.563 | 3.616, 2.143, 2.397, 2.150, 2.547, 2.224
python entry to GUI main start | 6 | 0 | 63.053 | 63.293 | 61.151 | 66.646 | 66.646 | 2.099 | 62.399, 61.151, 63.707, 66.646, 61.355, 64.503
runtime import | 6 | 0 | 880.897 | 877.324 | 865.205 | 884.306 | 884.306 | 8.221 | 883.636, 869.005, 882.861, 878.933, 865.205, 884.306
config load | 6 | 0 | 42.173 | 45.989 | 6.001 | 99.805 | 99.805 | 44.189 | 6.001, 99.805, 77.651, 6.129, 79.650, 6.696
first background load | 6 | 0 | 1696.915 | 1737.295 | 1672.333 | 1903.067 | 1903.067 | 87.498 | 1688.220, 1903.067, 1702.708, 1672.333, 1766.320, 1691.123
first simulation compute | 6 | 0 | 1090.379 | 1407.280 | 1054.185 | 2093.805 | 2093.805 | 517.609 | 2056.252, 2093.805, 1095.887, 1054.185, 1084.871, 1058.678
first GUI render after compute | 6 | 0 | 479.590 | 493.904 | 327.095 | 851.158 | 851.158 | 191.654 | 464.474, 851.158, 497.382, 327.095, 494.705, 328.611
draw event to Tk idle | 6 | 0 | 18.514 | 21.772 | 9.286 | 44.151 | 44.151 | 13.705 | 15.149, 30.872, 44.151, 9.293, 21.878, 9.286

## Startup By Group

### fresh_process

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
process launch to first visible | 5 | 0 | 6595.999 | 7522.914 | 6220.233 | 9748.664 | 9748.664 | 1613.860 | 8730.667, 9748.664, 6319.005, 6220.233, 6595.999
process launch overhead | 5 | 0 | 2.397 | 2.571 | 2.143 | 3.616 | 3.616 | 0.609 | 3.616, 2.143, 2.397, 2.150, 2.547
python entry to GUI main start | 5 | 0 | 62.399 | 63.052 | 61.151 | 66.646 | 66.646 | 2.251 | 62.399, 61.151, 63.707, 66.646, 61.355
runtime import | 5 | 0 | 878.933 | 875.928 | 865.205 | 883.636 | 883.636 | 8.358 | 883.636, 869.005, 882.861, 878.933, 865.205
config load | 5 | 0 | 77.651 | 53.847 | 6.001 | 99.805 | 99.805 | 44.471 | 6.001, 99.805, 77.651, 6.129, 79.650
first background load | 5 | 0 | 1702.708 | 1746.530 | 1672.333 | 1903.067 | 1903.067 | 94.500 | 1688.220, 1903.067, 1702.708, 1672.333, 1766.320
first simulation compute | 5 | 0 | 1095.887 | 1477.000 | 1054.185 | 2093.805 | 2093.805 | 546.298 | 2056.252, 2093.805, 1095.887, 1054.185, 1084.871
first GUI render after compute | 5 | 0 | 494.705 | 526.963 | 327.095 | 851.158 | 851.158 | 194.210 | 464.474, 851.158, 497.382, 327.095, 494.705
draw event to Tk idle | 5 | 0 | 21.878 | 24.269 | 9.293 | 44.151 | 44.151 | 13.712 | 15.149, 30.872, 44.151, 9.293, 21.878

### warm_fresh_process

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
process launch to first visible | 1 | 0 | 6720.759 | 6720.759 | 6720.759 | 6720.759 | 6720.759 | 0.000 | 6720.759
process launch overhead | 1 | 0 | 2.224 | 2.224 | 2.224 | 2.224 | 2.224 | 0.000 | 2.224
python entry to GUI main start | 1 | 0 | 64.503 | 64.503 | 64.503 | 64.503 | 64.503 | 0.000 | 64.503
runtime import | 1 | 0 | 884.306 | 884.306 | 884.306 | 884.306 | 884.306 | 0.000 | 884.306
config load | 1 | 0 | 6.696 | 6.696 | 6.696 | 6.696 | 6.696 | 0.000 | 6.696
first background load | 1 | 0 | 1691.123 | 1691.123 | 1691.123 | 1691.123 | 1691.123 | 0.000 | 1691.123
first simulation compute | 1 | 0 | 1058.678 | 1058.678 | 1058.678 | 1058.678 | 1058.678 | 0.000 | 1058.678
first GUI render after compute | 1 | 0 | 328.611 | 328.611 | 328.611 | 328.611 | 328.611 | 0.000 | 328.611
draw event to Tk idle | 1 | 0 | 9.286 | 9.286 | 9.286 | 9.286 | 9.286 | 0.000 | 9.286

## Theta Change To 10 Deg

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
input to var write | 0 | 0 |  |  |  |  |  |  | 
variable write to update scheduled | 0 | 0 |  |  |  |  |  |  | 
update queue delay | 0 | 0 |  |  |  |  |  |  | 
simulation calculation only | 0 | 0 |  |  |  |  |  |  | 
GUI update after calculation | 0 | 0 |  |  |  |  |  |  | 
result ready to canvas draw complete | 0 | 0 |  |  |  |  |  |  | 
canvas draw complete to Tk idle visible | 0 | 0 |  |  |  |  |  |  | 
total change to visible | 0 | 0 |  |  |  |  |  |  | 
total excluding automated typing | 0 | 0 |  |  |  |  |  |  | 

## Theta Return To Original

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
input to var write | 0 | 0 |  |  |  |  |  |  | 
variable write to update scheduled | 0 | 0 |  |  |  |  |  |  | 
update queue delay | 0 | 0 |  |  |  |  |  |  | 
simulation calculation only | 0 | 0 |  |  |  |  |  |  | 
GUI update after calculation | 0 | 0 |  |  |  |  |  |  | 
result ready to canvas draw complete | 0 | 0 |  |  |  |  |  |  | 
canvas draw complete to Tk idle visible | 0 | 0 |  |  |  |  |  |  | 
total change to visible | 0 | 0 |  |  |  |  |  |  | 
total excluding automated typing | 0 | 0 |  |  |  |  |  |  | 

## Redraw Only

metric | trial_count | missing_count | median_ms | mean_ms | min_ms | max_ms | p95_ms | std_ms | raw_ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
input to visible | 0 | 0 |  |  |  |  |  |  | 
rasterize | 0 | 0 |  |  |  |  |  |  | 
display set_data | 0 | 0 |  |  |  |  |  |  | 
canvas draw | 0 | 0 |  |  |  |  |  |  | 
canvas draw complete to Tk idle visible | 0 | 0 |  |  |  |  |  |  | 

- Redraw updates visible: `0`; compute_absent events: `0`; compute spans during redraw: `0`.

## Raw Theta Rows

No theta-change rows captured.

## Trial Results

artifact | measured | returncode | timed_out | termination | child_pid
--- | --- | --- | --- | --- | ---
trial_001.jsonl | True | 0 | False | normal | 38504
trial_002.jsonl | True | 0 | False | normal | 4020
trial_003.jsonl | True | 0 | False | normal | 35192
trial_004.jsonl | True | 0 | False | normal | 41096
trial_005.jsonl | True | 0 | False | normal | 30972
warmup_001.jsonl | False | 0 | False | normal | 41160
trial_006.jsonl | True | 0 | False | normal | 38144

## Notes

- Raw trial JSONL files contain the source-of-truth `perf_counter_ns` events.
- Local paths are redacted to basenames in metadata and timing event helpers.
- External pyautogui-style input is not run unless added separately; in-app Tk automation is used by default.