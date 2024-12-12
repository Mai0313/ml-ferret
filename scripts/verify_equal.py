"""Usage:
python3 misc/verify_equal.py \
    --orig-model-path ./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --new-model-path ./model/ferret-7b-v1-3

"""

import argparse

from tqdm import tqdm
import torch

from ferret import FERRETLlamaForCausalLM


def verify_equal(old_model_path, new_model_path) -> bool:
    old = FERRETLlamaForCausalLM.from_pretrained(
        old_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    new = FERRETLlamaForCausalLM.from_pretrained(
        new_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    # Get state dictionaries of both models
    state_dict1 = old.state_dict()
    state_dict2 = new.state_dict()

    # Compare each parameter
    for name, param in tqdm(state_dict1.items(), desc="Traverse all params"):
        # Check if the parameter name exists in the second model
        if name not in state_dict2:
            return False

        # Check if the parameter weights are the same, bf16 vs. f32
        if not torch.allclose(param, state_dict2[name], atol=1e-4):
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-model-path", type=str, required=True)
    parser.add_argument("--new-model-path", type=str, required=True)

    args = parser.parse_args()
