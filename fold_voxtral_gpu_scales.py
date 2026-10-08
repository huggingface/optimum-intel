import json
import os
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

source_dir = Path("/home/krz/models/voxtral-realtime-2602")
output_dir = Path("/home/krz/models/voxtral-realtime-2602-gpufix-delay2")
delay = 2

source_decoder = source_dir / "openvino_text_decoder.xml"
assert source_decoder.is_file(), source_decoder
assert not output_dir.exists(), f"Refusing to overwrite {output_dir}"

output_dir.mkdir()
for name in (
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tekken.json",
    "openvino_audio_encoder.xml",
):
    shutil.copy2(source_dir / name, output_dir / name)

config = json.loads((source_dir / "config.json").read_text())
config["default_num_delay_tokens"] = delay
(output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
os.link(source_dir / "openvino_audio_encoder.bin", output_dir / "openvino_audio_encoder.bin")

core = ov.Core()
model = core.read_model(source_decoder)
nodes = {node.get_friendly_name(): node for node in model.get_ordered_ops()}

scale_outputs = []
for layer in range(64):
    node = nodes.get(
        f"__module.language_model.model.layers.{layer}/aten::add/Add_1"
    )
    if node is None:
        break
    assert list(node.get_output_partial_shape(0)) == [1, 3072]
    scale_outputs.append(node.output(0))

assert len(scale_outputs) == 26, f"Expected 26 scales, found {len(scale_outputs)}"

delay_input = [
    parameter
    for parameter in model.get_parameters()
    if parameter.get_friendly_name() == "num_delay_tokens"
]
assert len(delay_input) == 1, f"Expected one num_delay_tokens input, found {len(delay_input)}"

probe = ov.Model(scale_outputs, delay_input, "ada_scales")
compiled = core.compile_model(
    probe,
    "CPU",
    {"INFERENCE_PRECISION_HINT": "f32"},
)
values = compiled(
    {"num_delay_tokens": np.array([float(delay)], dtype=np.float32)}
)

for index, output in enumerate(scale_outputs):
    scale = np.asarray(values[compiled.outputs[index]], dtype=np.float32)
    output.replace(ops.constant(scale, ov.Type.f32).output(0))

model.validate_nodes_and_infer_types()
ov.save_model(
    model,
    output_dir / "openvino_text_decoder.xml",
    compress_to_fp16=False,
)
print(f"Folded {len(scale_outputs)} scales for num_delay_tokens={delay}")
