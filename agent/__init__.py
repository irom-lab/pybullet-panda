from .agent_grasp import AgentGrasp
from .agent_grasp_collect import AgentGraspCollect
from .agent_grasp_mv import AgentGraspMV
from .agent_push import AgentPush
from .agent_grasp_eq import AgentGraspEq


def get_agent(name):
    if name == 'AgentGrasp':
        return AgentGrasp
    elif name == 'AgentGraspCollect':
        return AgentGraspCollect
    elif name == 'AgentGraspMV':
        return AgentGraspMV
    elif name == 'AgentGraspEq':
        return AgentGraspEq
    elif name == 'AgentPush':
        return AgentPush
    else:
        raise 'Unknown agent type!'
