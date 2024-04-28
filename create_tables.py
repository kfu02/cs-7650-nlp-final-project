import math
import re
import pprint
import sys

def read_file(input_filename):
    with open(input_filename, 'r') as file:
        raw_log = file.read()
        log = re.sub(r'\s+', '', raw_log)
    return log

def remove_tensor_info(raw_str):
    return re.sub(r'tensor\(|,?\s*device=\'cuda:0\'\)', '', raw_str)

def extract_tensors(log):
    # Regular expression pattern to match tensors
    tensor_pattern = r'tensor\(\[.*?\)'

    # Find all tensor matches in the log content
    tensor_matches = re.findall(tensor_pattern, log)

    all_tensors = []
    for i, tensor in enumerate(tensor_matches, start=1):
        all_tensors.append(eval(remove_tensor_info(tensor)))

    return all_tensors

def convert_to_dict(all_tensors):
    # every group of 4 tensors belongs to the same iter
    output = {}
    for i in range(len(all_tensors) // 4):
        tensor_i = i * 4
        edge_indices = all_tensors[tensor_i]
        edge_weights = all_tensors[tensor_i+1]
        obs = all_tensors[tensor_i+2]
        node_weights = all_tensors[tensor_i+3]

        # we only care about edge weights and obs, since edge_indices are fixed and node_weights are irrelevant
        output[i] = {'edge_weights': edge_weights, 'obs': obs}

    return output

def create_dict_from_log(input_filename):
    log = read_file(input_filename)

    all_tensors = extract_tensors(log)
    print(f'Num tensors: {len(all_tensors)}')

    return convert_to_dict(all_tensors)

def compute_interagent_dist(obs1, obs2):
    """
    Given observations from two agents, compute their distance

    obs: [p_x, p_y, v_x, v_y, rel_goal_x, rel_goal_y]
    """
    x1 = obs1[0]
    y1 = obs1[1]

    x2 = obs2[0]
    y2 = obs2[1]

    ROBOT_RADIUS = 0.1

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - 2*ROBOT_RADIUS

def find_all_dists(all_obs):
    """
    Given observations from all 3 agents, compute all interagent dists

    obs: [p_x, p_y, v_x, v_y, rel_goal_x, rel_goal_y]
    """
    r0_r1_dist = compute_interagent_dist(all_obs[0], all_obs[1])
    r1_r2_dist = compute_interagent_dist(all_obs[1], all_obs[2])
    r0_r2_dist = compute_interagent_dist(all_obs[0], all_obs[2])
    return r0_r1_dist, r1_r2_dist, r0_r2_dist

def find_all_vels(all_obs):
    """
    Given observations from all 3 agents, find all velocities

    obs: [p_x, p_y, v_x, v_y, rel_goal_x, rel_goal_y]
    """
    r0_vel = all_obs[0][2:4]
    r1_vel = all_obs[1][2:4]
    r2_vel = all_obs[2][2:4]

    return r0_vel, r1_vel, r2_vel

def find_all_rel_goals(all_obs):
    """
    Given observations from all 3 agents, find all goal positions

    obs: [p_x, p_y, v_x, v_y, rel_goal_x, rel_goal_y]
    """
    # TODO: replace with dist to goal?
    r0_goal = all_obs[0][-2:]
    r1_goal = all_obs[1][-2:]
    r2_goal = all_obs[2][-2:]

    return r0_goal, r1_goal, r2_goal

def generate_filtered_table(input_filename, output_filename):
    """
    Where filtered means instead of raw obs for each agent, we directly input the agent-agent distances.
    # TODO: also try True/False for attn?
    # TODO: also try avging attn
    """
    log_dict = create_dict_from_log(input_filename)
    # pprint.pp(log_dict, width=120)
    semistructured_table = "timestep | " + \
                           "r0 goal | r1 goal | r2 goal | " + \
                           "r0 vel | r1 vel | r2 vel | " + \
                           "r0-r1 attention | r0-r2 attention | r1-r0 attention | r1-r2 attention | r2-r0 attention | r2-r1 attention | " + \
                           "r0-r1 distance | r0-r2 distance | r1-r2 distance\n"

    for iter, info in log_dict.items():
       row = f"{iter} | "
       all_obs = info['obs']

       for goal in find_all_rel_goals(all_obs):
           # TODO: split into x/y?
           row += f"{goal} | "

       for vel in find_all_vels(all_obs):
           # TODO: split into x/y?
           row += f"{vel} | "

       for edge_weight in info['edge_weights']:
           row += f"{edge_weight} | "

       for dist in find_all_dists(all_obs):
           row += f"{dist} | "

       semistructured_table += row + "\n"

    with open(output_filename, 'w') as f:
        f.write(semistructured_table)

def generate_raw_obs_table(input_filename, output_filename):
    log_dict = create_dict_from_log(input_filename)
    # pprint.pp(log_dict, width=120)
    semistructured_table = "timestep | " + \
                           "r0-r1 attention | r0-r2 attention | r1-r0 attention | r1-r2 attention | r2-r0 attention | r2-r1 attention | " + \
                           "r0 observations | r1 observations | r2 observations\n"

    for iter, info in log_dict.items():
       row = f"{iter} | "
       for edge_weight in info['edge_weights']:
           row += f"{edge_weight} | "
       for obs in info['obs']:
           row += f"{obs} | "
       semistructured_table += row + "\n"

    with open(output_filename, 'w') as f:
        f.write(semistructured_table)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    generate_raw_obs_table(input_filename, f"{input_filename}-raw_obs_table.txt")
    generate_filtered_table(input_filename, f"{input_filename}-filtered_table.txt")

if __name__ == "__main__":
    main()
