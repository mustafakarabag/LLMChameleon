import os
import torch
import numpy as np
import unicodedata
import re
from functools import wraps
import time


def dict_to_str(d, as_filename=False):
    def print_val(v):
        if type(v) == dict:
            return dict_to_str(v)
        elif type(v) in (list, tuple):
            return [print_val(_v) for _v in v]
        elif type(v) == str or v == None:
            return v
        else:
            return f"{v:.3f}"
    if as_filename:
        return "___".join([f"{k}={print_val(v)}" for k, v in d.items()])
    else:
        return ", ".join([f"{k}: {print_val(v)}" for k, v in d.items()])


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def seed_all(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # multi-GPU.

        ### deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def timeit(func):
    """
    Source: https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"> {func.__name__}(): {total_time / 60:.2f} mins = {total_time:.2f} secs = {total_time * 1000:.2f} ms")
        return result
    return timeit_wrapper


def prepare_dict_for_saving(cfg_dict):
    """Prepare the config dictionary for saving, keeps only keys that contain primitive types (+ standard collections)."""
    def prepare_value_for_saving(value):
        """Recursively prepare values for saving."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, np.ndarray)):
            return [prepare_value_for_saving(item) for item in value]
        elif isinstance(value, dict):
            return prepare_dict_for_saving(value)
        elif isinstance(value, tuple):
            return tuple(prepare_value_for_saving(item) for item in value)
        elif isinstance(value, set):
            return {prepare_value_for_saving(item) for item in value}
        elif isinstance(value, type):
            return value.__name__
        else:
            return None
    
    prepared_dict = {}
    for key, value in cfg_dict.items():
        prepared_value = prepare_value_for_saving(value)
        if value is not None and prepared_value is None:
            prepared_dict[key] = f"Unsupported type: {type(value).__name__} for key: {key}"
        else:
            prepared_dict[key] = prepared_value
    return prepared_dict
