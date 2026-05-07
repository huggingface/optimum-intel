import torch

from optimum.exporters.openvino.model_patcher import _create_causal_mask_ov_compatible


class _TraceWrapper(torch.nn.Module):
    def forward(self, inputs_embeds, attention_mask, position_ids):
        return _create_causal_mask_ov_compatible(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


def test_qwen3_5_causal_mask_trace_avoids_aten_index():
    inputs_embeds = torch.zeros(1, 17, 8)
    attention_mask = torch.ones(1, 17, dtype=torch.bool)
    position_ids = torch.arange(17, dtype=torch.long).view(1, 1, 17).expand(4, 1, 17)

    traced = torch.jit.trace(_TraceWrapper(), (inputs_embeds, attention_mask, position_ids))
    graph_text = str(traced.graph)
    mask = traced(inputs_embeds, attention_mask, position_ids)

    assert mask.shape == (1, 1, 17, 17)
    assert "aten::index" not in graph_text
    assert mask[0, 0, 0, 0].item() == 0
    assert mask[0, 0, 0, 1].item() < 0
