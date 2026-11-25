"""Independent arithmetic checks for operators that generic MAC tools miss."""

import torch
from torch import nn
import torch.nn.functional as F

from cscwd_router.code.stag.complexity import flop_counter


def test_fused_cpu_attention_includes_both_matrix_products():
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 11, 8)
    value = torch.randn(2, 3, 11, 5)
    with torch.inference_mode(), flop_counter() as counter:
        F.scaled_dot_product_attention(query, key, value)
    assert counter.get_total_flops() == 2 * 2 * 3 * 7 * 11 * (8 + 5)


def test_fused_equal_width_cpu_attention_is_not_silently_omitted():
    value = torch.randn(1, 2, 17, 8)
    with torch.inference_mode(), flop_counter() as counter:
        F.scaled_dot_product_attention(value, value, value)
    assert counter.get_total_flops() == 2 * 1 * 2 * 17 * 17 * (8 + 8)


def test_transposed_convolution_uses_input_grid_for_mac_count():
    model = nn.ConvTranspose2d(3, 5, 2, stride=2, bias=False)
    value = torch.randn(2, 3, 7, 11)
    with torch.inference_mode(), flop_counter() as counter:
        model(value)
    assert counter.get_total_flops() == 2 * 2 * 7 * 11 * 3 * 5 * 2 * 2
