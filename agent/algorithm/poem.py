import torch
import numpy as np
import torch.nn.functional as F

EPS = 1e-9 # epsilon global variable for similarity matrix


def metric_fixed_point(cost_matrix, gamma=0.99, device='cpu'): # fast version of metric fixed point calclulation
  """Dynamic programming for calculating PSM (for alignment loss)."""
  d = torch.zeros_like(cost_matrix) #[56, 56]
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if torch.sum(torch.abs(d - d_new)) < EPS:
      break
    else:
      d = d_new[:]
  return d


def calculate_action_cost_matrix(actions_1, actions_2, hardcode_encoder):
    """ Action cost matrix calculation for alignment loss"""
    if hardcode_encoder:
        action_equality = torch.eq(actions_1.unsqueeze(1), actions_2.unsqueeze(0)).float()	# boolean to float for identity
        return 1.0 - action_equality # for identity encoder
    else: 
        action_equality = torch.cdist(actions_1, actions_2, p=1) #  for non-identity 
        return action_equality # for non-identity 


def calculate_reward_cost_matrix(rewards_1, rewards_2):
    """ Reward cost matrix calculation for alignment loss """
    diff = rewards_1.unsqueeze(1) - rewards_2.unsqueeze(0)
    return torch.abs(diff)


# Helper functions for alignment loss calculations
def gather_row(tensor, indices):
    num_row = tensor.shape[0]
    new_tensor = torch.zeros((num_row)).to(tensor.device)
    for i in range(num_row):
        new_tensor[i] = tensor[i, indices[i]]
    return new_tensor


def gather_col(tensor, indices):
    num_col = tensor.shape[1]
    new_tensor = torch.zeros((num_col)).to(tensor.device)
    for i in range(num_col):
        new_tensor[i] = tensor[indices[i], i]
    return new_tensor


def update_row(tensor, indices, new_vals):
    num_row = tensor.shape[0]
    for i in range(num_row):
        tensor[i, indices[i]] = new_vals[i]
    return tensor


def update_col(tensor, indices, new_vals):
    num_col = tensor.shape[1]
    for i in range(num_col):
        tensor[indices[i], i] = new_vals[i]
    return tensor


def cosine_similarity(a, b, eps=1e-8):
    """ Added eps for numerical stability """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def ground_truth_coupling(actions_1, actions_2):
    """Calculates ground truth coupling using optimal actions on two envs."""
    diff = actions_2.index(1) - actions_1.index(1)
    assert diff >= 0, 'Please pass the actions_2 as actions_1 and vice versa!'
    n, m = len(actions_1), len(actions_2)
    cost_matrix = np.ones((n, m), dtype=np.float32)
    for i in range(n):
        j = i + diff
        if j < m:
            cost_matrix[i, j] = 0.0
        else:
            break
    return cost_matrix


def contrastive_loss(similarity_matrix,
                        metric_values,
                        temperature,
                        coupling_temperature=1.0,
                        use_coupling_weights=True):
    """Contrative Loss with soft coupling (Used for alignment loss) """

    similarity_matrix /= temperature
    neg_logits1, neg_logits2 = similarity_matrix.clone(), similarity_matrix.clone()
    col_indices = torch.argmin(metric_values, dim=1)
    pos_logits1 = gather_row(similarity_matrix, col_indices)
    row_indices = torch.argmin(metric_values, dim=0)
    pos_logits2 = gather_col(similarity_matrix, row_indices)

    # normalize metric_values and multiply by 1/coupling_temperature to use soft coupling
    if use_coupling_weights:
        metric_values = F.normalize(metric_values.view(1,-1), p=2, dim=1).reshape(metric_values.shape)*100
    else:
        metric_values = F.normalize(metric_values.view(1,-1), p=2, dim=1).reshape(metric_values.shape)

    if use_coupling_weights:
        # metric_values /= coupling_temperature
        coupling = torch.exp(-metric_values)  #Gamma(x,y)
        pos_weights1 = -gather_row(metric_values, col_indices)
        pos_weights2 = -gather_col(metric_values, row_indices)
        pos_logits1 += pos_weights1
        pos_logits2 += pos_weights2
        negative_weights = torch.log((1.0 - coupling) + EPS)
        neg_logits1 += update_row(negative_weights, col_indices, pos_weights1)
        neg_logits2 += update_col(negative_weights, row_indices, pos_weights2)
    neg_logits1 = torch.logsumexp(neg_logits1, dim=1) 
    neg_logits2 = torch.logsumexp(neg_logits2, dim=0)

    loss1 = torch.mean(neg_logits1 - pos_logits1)
    loss2 = torch.mean(neg_logits2 - pos_logits2)
    return loss1 + loss2, col_indices, row_indices


def representation_alignment_loss(state_encoder,
                                action_encoder,
                                optimal_data_tuple,
                                learned_encoder_coefficient,
                                hardcode_encoder=False,
                                use_bisim=False,
                                gamma=0.999,
                                use_l2_loss=False,
                                use_coupling_weights=False,
                                coupling_temperature=1.0,
                                temperature=1.0,
                                ground_truth=False,
                                device='cpu'):

    """Representation alignment loss."""
    obs_1, actions_1, rewards_1 = optimal_data_tuple[0]
    obs_2, actions_2, rewards_2 = optimal_data_tuple[1]

    # Convert all to tensors and push to device
    obs_1 = torch.from_numpy(obs_1).float().to(device).permute(0,3,1,2)
    obs_2 = torch.from_numpy(obs_2).float().to(device).permute(0,3,1,2)
    actions_1 = torch.tensor(actions_1).int().to(device)
    actions_2 = torch.tensor(actions_2).int().to(device)
    rewards_1 = torch.tensor(rewards_1).float().to(device)
    rewards_2 = torch.tensor(rewards_2).float().to(device)

    representation_1 = state_encoder.representation(obs_1)
    representation_2 = state_encoder.representation(obs_2)

    actions_abstract_1, actions_abstract_1_normalized = action_encoder(obs_1, actions_1, learned_encoder_coefficient)
    actions_abstract_2, actions_abstract_2_normalized = action_encoder(obs_2, actions_2, learned_encoder_coefficient)

    if use_l2_loss:
        similarity_matrix = torch.cdist(representation_1, representation_2, p=2)
    else:
        similarity_matrix = cosine_similarity(representation_1, representation_2)

    # to check against other losses
    # baselines - Elise's (plannable approximations), Rishab's, DBC
    if ground_truth:
        metric_vals = torch.tensor(
            ground_truth_coupling(actions_abstract_1, actions_abstract_2), dtype=torch.float32)
    elif use_bisim:
        cost_matrix = calculate_reward_cost_matrix(rewards_1, rewards_2)
    else:
        cost_matrix = calculate_action_cost_matrix(actions_abstract_1_normalized, actions_abstract_2_normalized, hardcode_encoder)
        metric_vals = metric_fixed_point(cost_matrix, gamma, device=device)
    if use_l2_loss:
        # Directly match the l2 distance between representations to metric values
        alignment_loss = torch.mean((similarity_matrix - metric_vals)**2)
    else:
        alignment_loss = contrastive_loss(similarity_matrix,
                                            metric_vals,
                                            temperature,
                                            coupling_temperature=coupling_temperature,
                                            use_coupling_weights=use_coupling_weights)

    return alignment_loss, metric_vals, similarity_matrix
