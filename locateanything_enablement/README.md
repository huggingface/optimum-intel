# LocateAnything-3B OpenVINO enablement — eval & parity scripts

Reusable scripts used to validate the LocateAnything-3B OpenVINO export
(FP16/INT8/INT4) on CPU + GPU. Run them from the workspace root
`dev/nvidia-LocateAnything-3B/` (paths to `ov_ir_*`, `parity_artifacts/`,
`eval_images/`, `eval_artifacts/` are relative to that dir).

| script | purpose |
| :--- | :--- |
| `ref_pytorch.py` | PyTorch reference: prefill logits + greedy slow-AR artifacts (Milestone B). |
| `parity.py` | Milestone-B FP16 OV-vs-PyTorch parity (vision/prefill cossim + token match). |
| `parity_multi.py` | Same parity generalized over `--ir-dir {fp16,int8,int4}` × `--device {CPU,GPU.0,...}`. |
| `eval_box_parity.py` | Box IoU / F1 accuracy parity (OV vs PyTorch GT): F1@0.5, F1@mean, corner-L1, box-count. |
| `perf.py` | Informational slow-AR LLM TTFT / ITL across precision × device. |

Reproduce (in the dedicated venv, with HF_TOKEN sourced):

```bash
./venv/bin/python ref_pytorch.py
./venv/bin/python parity_multi.py --ir-dir ov_ir_int4 --device CPU
./venv/bin/python eval_box_parity.py --precisions fp16,int8,int4 --devices CPU,GPU.0
./venv/bin/python perf.py --precisions fp16,int4 --devices CPU,GPU.0 --runs 3
```

Quantized IRs are produced via:
```bash
optimum-cli export openvino --model nvidia/LocateAnything-3B --task image-text-to-text \
    --trust-remote-code --weight-format int8 ov_ir_int8
optimum-cli export openvino --model nvidia/LocateAnything-3B --task image-text-to-text \
    --trust-remote-code --weight-format int4 --group-size 128 --ratio 1.0 ov_ir_int4
```
