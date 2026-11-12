# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
import functools
import os
import os.path as osp
import time
import argparse
from shutil import copyfile
from typing import Union
import numpy as np
import torch
from mmengine import Config, MMLogger, print_log
from mmengine.config import DictAction
from mmengine.utils import mkdir_or_exist
log_to_current = functools.partial(print_log, logger='current')

def parse_mmengine_args(override_mode="default"):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', help='config file path')
    args = None

    if override_mode == "default":
        """
        Use case:
            args = parser.parse_args(["config", "--cfg-options", "data=1", "flag=false"])
            -> Namespace(cfg_options={'data': 1, 'flag': False}, config='config')
        """
        parser.add_argument("--cfg-options", nargs='+', action=DictAction)
        args = parser.parse_args()
    elif override_mode == "key-value":
        """
        Use case:
            args, unknown = parser.parse_known_args(["config", "--data", "1", "--flag", "false"])
            ...
            -> Namespace(cfg_options={'data': 1, 'flag': False}, config='config')
        """
        args, unknown = parser.parse_known_args()
        assert len(unknown) % 2 == 0
        cfg_options = {}
        while len(unknown) != 0:
            k = unknown.pop(0)
            v = unknown.pop(0)
            assert k.startswith("--") and not v.startswith("--")
            cfg_options[k[2:]] = DictAction._parse_iterable(v)
        args.cfg_options = cfg_options

    return args


def flatten_nested_dict(nested: Union[dict, Config]) -> dict:
    """
        Inputs: 
            {"a": {"b": 1}, "c": 2} 
        Returns:
            {"a.b": 1, "c": 2}
    """
    stack = list(nested.items())
    ans = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}.{sub_key}", sub_val))
        else:
            ans[key] = val
    return ans


def init_mmengine_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        flatten_cfg = flatten_nested_dict(cfg)
        for ele in args.cfg_options.keys():
            if ele not in flatten_cfg:
                raise Exception(f"{ele} not in config!")
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def _get_next_version(root_dir, dir_name_prefix):
    dir_list = [d for d in os.listdir(root_dir) if d.startswith(dir_name_prefix)]

    new_id = 0

    if len(dir_list) > 0:
        exist_ids = [int(i.split('_')[-1]) for i in dir_list if len(i.split('_')) > 0 and i.split('_')[-1].isdigit()]
        if len(exist_ids) > 0:
            new_id = max(exist_ids) + 1
    return new_id


def pl_init_exp(override_mode="default",
                exp_prefix='',
                backup_list=None,
                inplace=False,
                work_dir_name="work_dirs",
                project_name="cryosage"):
    args = parse_mmengine_args(override_mode=override_mode)
    cfg = init_mmengine_config(args)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if len(exp_prefix) and not exp_prefix.endswith('_'):
        exp_prefix += '_'

    if not hasattr(cfg, "exp_name") or cfg.exp_name == "":
        exp_name = exp_prefix + osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = cfg.exp_name
    cfg.work_dir = osp.join(work_dir_name, exp_name)

    # lightning.fabric.utilities.distributed._distributed_available is True until fit start
    if "WORLD_SIZE" in os.environ:
        _ = MMLogger.get_instance(name="MMLogger", logger_name=project_name + "_fork")
        return cfg

    if not inplace:
        if osp.exists(work_dir_name):
            version = _get_next_version(work_dir_name, exp_name)
            cfg.work_dir += f"_{version:d}"

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # backup scripts
    if backup_list is not None:
        for p in backup_list:
            copyfile(p, osp.join(cfg.work_dir, osp.basename(p)))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, "config.py"))
    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_logger(project_name, log_file=log_file)
    logger = MMLogger.get_instance(name="MMLogger", logger_name=project_name, log_file=log_file)
    logger.info("Config:\n" + cfg.pretty_text)
    return cfg


def pretty_dict(x, precision=3, sci_mode=False):
    """
        Input: 
            x:  a dict (containing torch.Tensor, np.ndarray, float, etc.) to show in a line
            precision:  the precision of float number

        Returns:
            a formatted string, following `fairseq`'s style

        Example:
        ```
            pretty_dict({  
                k1: 1, 
                k2: 1.414213562,
                k3: torch.Tensor(3.1415926535)
            }, 3)

            Output:
            k1 1 | k2 1.414 | k3 3.141
        ```
            
    """
    ret = None
    if isinstance(x, dict):
        ret = []
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                v = v.item()
            else:
                v = v
            if isinstance(v, float):
                # ret.append("{}: {:.3f}".format(k, v))
                if sci_mode:
                    float_formatter = "{}: {:.<PCS>e}".replace("<PCS>", str(precision))
                else:
                    float_formatter = "{}: {:.<PCS>f}".replace("<PCS>", str(precision))
                ret.append(float_formatter.format(k, v))
            else:
                ret.append("{}: {}".format(k, v))
        ret = " | ".join(ret)
    return ret


def create_sphere_mask(d, h, w, center=None, radius=None) -> np.ndarray:
    if center is None:
        center = (int(d / 2), int(h / 2), int(w / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    z, y, x = np.ogrid[:d, :h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    mask = dist_from_center <= radius
    return mask


def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

