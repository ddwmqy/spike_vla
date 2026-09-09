# Best GLUE Results

```latex
\textbf{Model} & \textbf{Time} & \textbf{MNLI}$_\text{m/mm}$ & \textbf{QQP}$_\text{F1}$ & \textbf{QNLI} & \textbf{SST-2} & \textbf{CoLA} & \textbf{STS-B} & \textbf{MRPC}$_\text{F1}$ & \textbf{RTE} & \textbf{Avg.}  & \textbf{Energy$_\text{mJ}$} \\
Ours & 4 & {75.8/76.0} & {84.7} & {84.6} & {90.1} & {50.4} & {82.4} & {87.9} & {61.7} & \textbf{77.1} & \textbf{5.79} \\
```

| Task | Metric | Best score | Epoch | Run | Log directory |
|---|---:|---:|---:|---|---|
| mnli | accuracy | 75.802/75.976 | 6 | run-20260125_035413-k1bs2vb5 | results/glue_logs/logs/mnli |
| qqp | f1 | 84.678 | 8 | run-20260125_001636-xga5ba87 | results/glue_logs/logs/qqp |
| qnli | accuracy | 84.569 | 9 | run-20260122_065331-e8fr3as5 | results/glue_logs/logs/qnli |
| sst2 | accuracy | 90.138 | 5 | run-20260122_104538-ehf49nyt | results/glue_logs/logs/sst2 |
| cola | matthews_correlation | 50.426 | 9 | run-20260120_091321-dyhr1x78 | results/glue_logs/logs/cola |
| stsb | spearmanr | 82.393 | 5 | run-20260122_140824-dw1hw6hm | results/glue_logs/logs/stsb |
| mrpc | f1 | 87.854 | 18 | run-20260119_230352-4f4ycuc7 | results/glue_logs/logs/mrpc |
| rte | accuracy | 61.733 | 11 | run-20260121_082922-j2nmlgrw | results/glue_logs/logs/rte |
