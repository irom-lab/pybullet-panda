# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

from warnings import simplefilter
import os
import argparse
import time
import pretty_errors
import matplotlib
import wandb
from shutil import copyfile

# AGENT
from agent.agent_grasp_mv import AgentGraspMV

# ENV
from panda_gym.grasp_mv_env import GraspMultiViewEnv
from panda_gym.vec_env_grasp_mv import VecEnvGraspMV
from alano.train.vec_env import make_vec_envs
from alano.utils.yaml import load_config

simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")
matplotlib.use('Agg')


def main(config_file, config_dict):
    # Config
    CONFIG_ENV = config_dict['environment']
    CONFIG_TRAINING = config_dict['training']
    CONFIG_ARCH = config_dict['arch']
    CONFIG_UPDATE = config_dict['update']
    os.makedirs(CONFIG_TRAINING.OUT_FOLDER, exist_ok=True)
    copyfile(config_file,
             os.path.join(CONFIG_TRAINING.OUT_FOLDER, 'config.yaml'))
    if CONFIG_TRAINING.USE_WANDB:
        wandb.init(entity='grasp-mv',
                   project=CONFIG_TRAINING.PROJECT_NAME,
                   name=CONFIG_TRAINING.NAME)
        wandb.config.update(CONFIG_ENV)
        wandb.config.update(CONFIG_TRAINING)

    # Environment
    print("\n== Environment Information ==")
    if CONFIG_ENV.ENV_NAME == 'GraspMultiView-v0':
        env_class = GraspMultiViewEnv
    else:
        raise NotImplementedError

    venv = make_vec_envs(
        env_name=CONFIG_ENV.ENV_NAME,
        seed=CONFIG_TRAINING.SEED,
        num_processes=CONFIG_TRAINING.NUM_CPUS,
        device=CONFIG_TRAINING.DEVICE,
        config_env=CONFIG_ENV,
        vec_env_type=VecEnvGraspMV,
        max_steps_train=CONFIG_ENV.MAX_TRAIN_STEPS,
        max_steps_eval=CONFIG_ENV.MAX_EVAL_STEPS,
        renders=False,  #!
        img_h=CONFIG_ENV.IMG_H,
        img_w=CONFIG_ENV.IMG_W,
        use_rgb=CONFIG_ENV.USE_RGB,
        use_depth=CONFIG_ENV.USE_DEPTH,
    )
    # env = env_class(
    #     max_steps_train=CONFIG_ENV.MAX_TRAIN_STEPS,
    #     max_steps_eval=CONFIG_ENV.MAX_EVAL_STEPS,
    #     use_append=CONFIG_ENV.USE_APPEND,
    #     obs_buffer=CONFIG_ENV.OBS_BUFFER,
    #     g_x_fail=CONFIG_ENV.G_X_FAIL,
    #     render=False,
    #     img_H=CONFIG_ENV.IMG_H,
    #     img_W=CONFIG_ENV.IMG_W,
    # )
    # venv.env_method('report', indices=[0])  # call the method for one env
    venv.reset()
    # env.reset()

    # Agent
    print("\n== Agent Information ==")
    if CONFIG_TRAINING.AGENT_NAME == 'AgentGraspMV':
        agent_class = AgentGraspMV
    agent = agent_class(CONFIG_TRAINING, CONFIG_ARCH, CONFIG_UPDATE,
                        CONFIG_ENV)
    print('\nTotal parameters in actor: {}'.format(
        sum(p.numel() for p in agent.performance.actor.parameters()
            if p.requires_grad)))
    print("We want to use: {}, and Agent uses: {}".format(
        CONFIG_TRAINING.DEVICE, agent.performance.device))

    # Learn
    print("\n== Learning ==")
    train_records, train_progress = agent.learn(venv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--config_file",
                        help="config file path",
                        type=str)
    args = parser.parse_args()
    config_dict = load_config(args.config_file)
    main(args.config_file, config_dict)
