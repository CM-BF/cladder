import pathlib
from dataclasses import dataclass
from jsonargparse import set_docstring_parse_options
import jsonargparse
from jsonargparse import ArgumentParser, ActionConfigFile
from argparse import ArgumentParser as origArgumentParser
import yaml
import wandb
from typing import List
from hashlib import sha1
import os

# import subprocess
# print(subprocess.check_output(['which', 'ptxas']))

def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 't', 'yes', 'y')
    raise ValueError("Input must be a string or a boolean")

def dump_config(cfg):
    yaml.add_representer(pathlib.Path, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    yaml.add_representer(pathlib.PosixPath,
                         lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    yaml.add_representer(jsonargparse._util.Path,
                         lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    return yaml.dump(cfg.as_dict(), sort_keys=True)



if __name__ == '__main__':
    # main()

    # --- pre parser ---
    pre_parser = origArgumentParser()
    pre_parser.add_argument('--config', type=pathlib.Path, help='Config file')
    pre_cfg, unk = pre_parser.parse_known_args()
    real_path = ROOT_DIR / pre_cfg.config
    simple_config = yaml.load(real_path.read_text(), Loader=yaml.FullLoader)
    experiment_name = simple_config['exp']['class_path']

    # --- main parser ---
    parser = ArgumentParser(parser_mode='omegaconf')
    # -- A wrapper for the experiment class: No nested key --
    parser.add_class_arguments(CEL, sub_configs=True)
    # parser.add_subclass_arguments(ExpClass, 'exp')
    if 'aphynity' in experiment_name.lower():
        parser.link_arguments('exp.dataloader.dataset', 'exp.init_args.net.init_args.dataset', apply_on='instantiate')
    # parser.link_arguments('exp.loader["train"].dataset.input_length', 'exp.init_args.model.init_args.input_length', apply_on='instantiate')
    parser.add_argument('--post_config', action=ActionConfigFile, help='Config file')
    NEW_RUN = to_bool(os.environ.get('NEW_RUN', True))
    if NEW_RUN:
        # parser.default_config_files = [str(real_path)]
        cfg = parser.parse_args(['--post_config', str(real_path)] + unk)
        # cfg = parser.parse_path(real_path)

        # cfg = parser.parse_args()
        dumped_config = dump_config(cfg)
        print(dumped_config)
        exp_hash = sha1(dumped_config.encode()).hexdigest()
        dict_cfg = cfg.as_dict()

        cfg = parser.instantiate_classes(cfg)
        if cfg.exp.wandb_logger:
            wandb.init(project='Invariant_Phy', name=eval(f"f{repr(cfg.name)}") if cfg.name is not None else None, tags=cfg.tags, config=dict_cfg | {'exp_hash': exp_hash})
        print(cfg.exp, '\nExperiment sha1:', exp_hash)
        # cfg.exp: ExpClass
        cfg.exp(exp_hash=exp_hash, dumped_config=dumped_config)