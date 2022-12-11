from .agent_grasp import AgentGrasp
from .agent_grasp_mv import AgentGraspMV
from .agent_push import AgentPush


def get_agent(name):
    if name == 'AgentGrasp':
        return AgentGrasp
    elif name == 'AgentGraspMV':
        return AgentGraspMV
    elif name == 'AgentPush':
        return AgentPush
    else:
        raise 'Unknown agent type!'
