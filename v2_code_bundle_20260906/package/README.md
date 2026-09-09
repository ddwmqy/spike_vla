# LIBERO SDT-V3 + BERT + SNN Fusion

迁移到其他服务器时先阅读 `RUNBOOK_CN.md`。正式入口是 `run_train.sh`，训练完成后的 clean LIBERO 评测入口是 `run_eval.sh`。

Student architecture:

- SDT-V3 19M vision encoder, trainable
- pretrained BERT language encoder, frozen
- 6-layer bidirectional SNN fusion, `T=4`
- exact Spike2Max normalization for the accuracy-oriented run
- learned softmax temporal readout over continuous fusion membrane/residual values
- ANN ACT decoder

Training uses the 80k SDT-V3+BERT checkpoint EMA in two ways:

- controlled warm-start of 603 vision, 70 ANN action-head, and 11 projection tensors
- feature/action distillation with weights `0.1/0.1`

The SNN fusion parameters are newly initialized. Training preserves the strong baseline recipe: frozen BERT, vision LR `1e-5`, head LR `5e-5`, weight decay `1e-10`, FP32, global batch 256, and checkpoints every 5k steps.

All modified Python is isolated under `code/`; the existing TurboVLA repositories are not changed.
