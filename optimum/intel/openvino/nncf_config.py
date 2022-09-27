from nncf import NNCFConfig

DEFAULT_QUANTIZATION_CONFIG = {
  "compression": {
    "algorithm": "quantization",
    "preset": "mixed",
    "overflow_fix": "disable",
    "initializer": {
      "range": {
        "num_init_samples": 300,
        "type": "mean_min_max"
      },
      "batchnorm_adaptation": {
        "num_bn_adaptation_samples": 0
      }
    },
    "scope_overrides": {
      "activations": {
        "{re}.*matmul_0": {
          "mode": "symmetric"
        }
      }
    },
    "ignored_scopes": [
      "{re}.*Embeddings.*",
      "{re}.*__add___[0-1]",
      "{re}.*layer_norm_0",
      "{re}.*matmul_1",
      "{re}.*__truediv__*"
    ]
  }
}

def get_config_with_input_info(json_config, model_input):
    input_info = [{'sample_size': list(value.shape), "type": "long"} for name, value in model_input.items()]
    json_config["input_info"] = input_info
    return NNCFConfig.from_dict(json_config)