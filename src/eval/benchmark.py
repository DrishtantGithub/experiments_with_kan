# src/eval/benchmark.py
"""
Simple benchmarking: param count, forward pass timing (single-batch), and FLOPs via ptflops.
"""
import time, torch
from src.eval.eval_metrics import count_params, get_flops_and_params
from src.utils.utils import get_device

def time_forward(model, input_tensor, n_runs=20):
    device = get_device()
    model.to(device)
    input_tensor = input_tensor.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_total = time.time() - t0
    return t_total / n_runs

def run_benchmark(model, input_res=(3,32,32), batch_size=32):
    device = get_device()
    # param count
    params = count_params(model)
    # flops
    try:
        macs, pf = get_flops_and_params(model, input_res, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        macs, pf = None, None
    import torch
    dummy = torch.randn(batch_size, *input_res)
    t = time_forward(model, dummy)
    return {'params': params, 'macs': macs, 'time_per_forward_s': t}
