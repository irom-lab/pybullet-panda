from .agent_train import AgentTrain
from .agent_imitate import AgentImitate
from .agent_collect import AgentCollect
from .agent_train_eq import AgentTrainEq


def get_agent(name):
    if name == 'AgentTrain':
        return AgentTrain
    elif name == 'AgentImitate':
        return AgentImitate
    elif name == 'AgentCollect':
        return AgentCollect
    elif name == 'AgentTrainEq':
        return AgentTrainEq
    else:
        raise 'Unknown agent type!'
