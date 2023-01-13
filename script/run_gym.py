import os
import sys
import argparse
import pretty_errors
import wandb
import matplotlib
# matplotlib.use('Agg')
from omegaconf import OmegaConf
import logging
import time

# AGENT
from agent import get_agent

# ENV
from panda_gym import get_env, get_vec_env, get_vec_env_cfg
from panda_gym.util.vec_env import make_vec_envs


def main(cfg, no_wandb=False):
    ################### Logging ###################
    log_file = os.path.join(cfg.out_folder, 'log.log')
    log_fh = logging.FileHandler(log_file, mode='w+')
    log_sh = logging.StreamHandler(sys.stdout)
    # log_format = '%(asctime)s %(levelname)s: %(message)s'
    log_format = '%(levelname)s: %(message)s'
    # Possible levels: DEBUG, INFO, WARNING, ERROR, CRITICAL    
    logging.basicConfig(format=log_format, level='INFO', 
        handlers=[log_sh, log_fh])

    ###################### cfg ######################
    os.makedirs(cfg.out_folder, exist_ok=True)
    if no_wandb:    # overwrite
        cfg.use_wandb = False
    if cfg.use_wandb:
        wandb.init(entity='allenzren',
                   project=cfg.project,
                   name=cfg.run)
        wandb.config.update(cfg)

    # Reuse some cfg
    cfg.policy.num_cpus = cfg.num_cpus
    cfg.policy.out_folder = cfg.out_folder
    cfg.policy.use_wandb = cfg.use_wandb
    cfg.policy.action_dim = cfg.env.action_dim
    cfg.policy.max_train_steps = cfg.env.max_train_steps
    cfg.policy.max_eval_steps = cfg.env.max_eval_steps

    # Learner
    if hasattr(cfg.env, 'action_dim'):
        cfg.policy.learner.arch.action_dim = cfg.env.action_dim
    cfg.policy.learner.eval = cfg.policy.eval
    if cfg.env.camera is not None:
        cfg.policy.learner.img_h = cfg.env.camera.img_h
        cfg.policy.learner.img_w = cfg.env.camera.img_w
        cfg.policy.learner.arch.img_h = cfg.env.camera.img_h
        cfg.policy.learner.arch.img_w = cfg.env.camera.img_w

    # Device
    cfg.policy.device = cfg.device
    cfg.policy.image_device = cfg.image_device
    cfg.policy.learner.device = cfg.device
    cfg.policy.utility.device = cfg.device

    # Use same seed
    cfg.policy.seed = cfg.seed
    cfg.policy.learner.seed = cfg.seed
    cfg.policy.utility.seed = cfg.seed

    ###################### Env ######################
    # Common args
    env_cfg = OmegaConf.create({'render': cfg.env.render,
                                'camera_param': cfg.env.camera})

    # Add
    if cfg.env.specific:
        env_cfg = OmegaConf.merge(env_cfg, cfg.env.specific)

    # Environment
    logging.info("== Environment Information ==")
    venv = make_vec_envs(
        env_type=get_env(cfg.env.name),
        seed=cfg.seed,
        num_process=cfg.num_cpus,
        cpu_offset=cfg.cpu_offset,
        device=cfg.device,
        vec_env_type=get_vec_env(cfg.env.name),
        vec_env_cfg=get_vec_env_cfg(cfg.env.name, cfg.env),
        **env_cfg,  # pass to individual env
    )

    # Agent
    logging.info("== Agent Information ==")
    agent = get_agent(cfg.agent)(cfg.policy, venv)
    # logging.info('Total parameters in policy: {}'.format(
    #     sum(p.numel() for p in agent.learner.parameters()
    #         if p.requires_grad)))
    logging.info("Using device: {}".format(cfg.device))

    # Learn
    start_time = time.time()
    if cfg.policy.eval:
        logging.info("\n== Evaluating ==")
        agent.evaluate()
    else:
        logging.info("\n== Learning ==")
        agent.learn(verbose=cfg.verbose)
    logging.info('\nTime used: {:.1f}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--cfg_file",
                        help="cfg file path",
                        type=str)
    parser.add_argument("-no_wb",
                        "--no_wandb",
                        action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    main(cfg, args.no_wandb)
