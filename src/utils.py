import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    """
    Source:
        https://github.com/LeCAR-Lab/dial-mpc/blob/main/dial_mpc/utils/io_utils.py#L15
    """
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        import numpy as np

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
    return dataclass(**kwargs)
