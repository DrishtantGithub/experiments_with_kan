# src/eval/eval_metrics.py
import torch
import numpy as np
from ptflops import get_model_complexity_info

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_flops_and_params(model, input_res=(3,32,32), device='cpu'):
    model.eval()
    with torch.cuda.device(0) if device=='cuda' else dummy_context():
        macs, params = get_model_complexity_info(model, input_res, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
    # ptflops returns MACs (multiply-adds) -> FLOPs ≈ 2 * MACs typically; we'll report MACs for consistency
    return macs, params

# small helper for cpu context (since get_model_complexity_info uses cuda context)
from contextlib import contextmanager
@contextmanager
def dummy_context():
    yield
