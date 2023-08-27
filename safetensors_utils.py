from torch import Tensor
import argparse
import torch
from safetensors.torch import save_file, load_file
def unflatten_dict(input_dict, separator="."):
    """
    Converts a reformatted dict to an original dict.
    
    Args:
        input_dict (dict): Dict has been reformatted to be converted.
        separator (str, optional): Delimiter between keys. Default is ".".
    
    Returns:
        dict: Dict.
    """
    output_dict = {}
    for key, value in input_dict.items():
        keys = key.split(separator)
        current_dict = output_dict
        for sub_key in keys[:-1]:
            if sub_key not in current_dict:
                current_dict[sub_key] = {}
            current_dict = current_dict[sub_key]
        current_dict[keys[-1]] = value
    return output_dict

def flatten_dict(input_dict, parent_key="", separator="."):
    """
    Converts a dict to a reformatted dict.
    
    Args:
        input_dict(dict): The original dict to convert.
        parent_key (str, optional): Key of the parent dict. Use for recursion. Default is ".".
        separator (str, optional): Separator between keys. Default is ".".
    
    Returns:
        dict: The dict has been reformatted.
    """
    items = []
    for key, value in input_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def to_safetensors(obj : object) -> dict[str, Tensor]:
    return flatten_dict(input_dict=obj)

def from_safetensors(tensors : dict[str, Tensor]):
    return unflatten_dict(tensors)

def save_tensors(obj: dict[str, Tensor], path, metadata):
    save_file(obj, path, metadata=metadata)

def load_tensors(path):
    return load_file(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("to_safetensors", action="store_true")

    parser.add_argument("--from_file", required=True)
    parser.add_argument("--to", required=True)

    args = parser.parse_args()

    if args.to_tensors == True:
        checkpoint = torch.load(args.from_file)
        safe_checkpoint = to_safetensors(checkpoint)
        for k, v in safe_checkpoint.items():
            print(k, type(v))