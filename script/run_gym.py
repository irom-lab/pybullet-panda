# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

from warnings import simplefilter
import os
import argparse
import time
import pretty_errors
import wandb
from shutil import copyfile

# AGENT
from agent.agent_grasp_mv import AgentGraspMV
from agent.agent_push import AgentPush

# ENV
from panda_gym.vec_env.vec_env import VecEnvGraspMV, VecEnvGraspMVRandom, VecEnvPush
from alano.train.vec_env import make_vec_envs
from alano.utils.yaml import load_config

simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")
import matplotlib
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
        wandb.init(entity='allenzren',
                   project=CONFIG_TRAINING.PROJECT_NAME,
                   name=CONFIG_TRAINING.NAME)
        wandb.config.update(CONFIG_ENV)
        wandb.config.update(CONFIG_TRAINING)

    # Environment
    print("\n== Environment Information ==")
    if CONFIG_ENV.ENV_NAME == 'GraspMultiView-v0':
        vec_env_type = VecEnvGraspMV
    elif CONFIG_ENV.ENV_NAME == 'GraspMultiViewRandom-v0':
        vec_env_type = VecEnvGraspMVRandom
    elif CONFIG_ENV.ENV_NAME == 'Push-v0':
        vec_env_type = VecEnvPush
    else:
        raise NotImplementedError
    venv = make_vec_envs(
        env_name=CONFIG_ENV.ENV_NAME,
        seed=CONFIG_TRAINING.SEED,
        num_processes=CONFIG_TRAINING.NUM_CPUS,
        device=CONFIG_TRAINING.DEVICE,
        config_env=CONFIG_ENV,
        vec_env_type=vec_env_type,
        renders=CONFIG_ENV.RENDER,
        use_rgb=CONFIG_ENV.USE_RGB,
        use_depth=CONFIG_ENV.USE_DEPTH,
        camera_params=CONFIG_ENV.CAMERA,
    )
    # venv.reset()

    # Agent
    print("\n== Agent Information ==")
    if CONFIG_TRAINING.AGENT_NAME == 'AgentGraspMV':
        agent_class = AgentGraspMV
    elif CONFIG_TRAINING.AGENT_NAME == 'AgentPush':
        agent_class = AgentPush
    else:
        raise NotImplementedError
    agent = agent_class(CONFIG_TRAINING, CONFIG_UPDATE, CONFIG_ARCH,
                        CONFIG_ENV)
    print('\nTotal parameters in actor: {}'.format(
        sum(p.numel() for p in agent.performance.actor.parameters()
            if p.requires_grad)))
    print("We want to use: {}, and Agent uses: {}".format(
        CONFIG_TRAINING.DEVICE, agent.performance.device))

    # Learn
    if CONFIG_TRAINING.EVAL:
        print("\n== Evaluating ==")
        agent.evaluate(venv)
    else:
        print("\n== Learning ==")
        train_records, train_progress = agent.learn(venv)


if __name__ == "__main__":
    import time
    s1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--config_file",
                        help="config file path",
                        type=str)
    args = parser.parse_args()
    config_dict = load_config(args.config_file)
    main(args.config_file, config_dict)
    print('Time used: ', time.time()-s1)
