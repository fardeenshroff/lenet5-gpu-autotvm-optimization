# Verified Tesla T4 five-epoch benchmark (2026-11-22)

**Origin:** User-exported Google Colab NVIDIA Tesla T4 experiment. Nine original JSON result objects were checked for internal consistency and saved in this directory. The original source export was a ZIP archive; these JSON representations preserve its numeric report fields.

| Metric | Colab CPU | Tesla T4 | Observed comparison |
|---|---:|---:|---:|
| Five-epoch wall time | 122.274 s | 82.904 s | **1.475× faster on GPU** |
| MNIST test accuracy, epoch 5 | 98.58% | 98.52% | 0.06 percentage-point difference |
| Median inference batch 1 | 0.4494 ms | 0.3486 ms | **1.29×** |
| Median inference batch 128 | 7.4287 ms | 0.4119 ms | **18.03×** |

**Correctness evidence:** CUDA matrix multiplication matched a CPU reference (maximum absolute error `4.1962e-05`); a separate C++/CUDA 5×5 convolution also passed its CPU comparison (maximum absolute error `9.53674e-07`). Its 0.00616-ms average CUDA-event timing is **kernel-only** and not comparable to model inference timings.

**Experiment specification:** Five PyTorch/Adam epochs, learning rate 0.001, batch size 256 for separately initialized CPU and GPU models. Inference compared **the same trained GPU model weights** on the host CPU and Tesla T4, with 30 warmups and 100 synchronized iterations at each batch size. Software: PyTorch 2.14.1+cu130, Python 3.13.16, Tesla T4 compute capability 7.5.

**Interpretive boundary:** These are PyTorch-vs-PyTorch device results, **not AutoTVM speedups**. No new TVM or AutoTVM kernel-schedule benchmark was performed. Single runs do not establish a statistical significance claim. Full methodology and references: [Research manuscript](../../paper/Batch_Size_Dependent_LeNet5_GPU_Study.md).
