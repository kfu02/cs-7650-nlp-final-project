import sys

def generate_table_example(table, interpretation):
    prompt = "Here is the input data you must analyze.\n\n"
    prompt += "Input table:\n"
    prompt += f"{table}\n"
    prompt += "Interpretation:"
    prompt += interpretation
    return prompt

def build_preamble():
    prompt = 'You will be presented with a table of data derived from a robot team where each robot has attempted to learn a policy to navigate to its goal without colliding with another robot. ' + \
            'This data is collected over a series of timesteps (denoted under the "timestep" column). ' + \
            '\n\n' + \
            'A value of 0.5 under "r0 dist to goal" means at that timestep, robot 0 is 0.5 m away from its goal. Robots have reached their goal when they are within 0.15 meters of the goal point. ' + \
            'A value of [-1.3, 0.2] under "r0 vel" means at that timestep, robot 0 is moving at a velocity of -1.3 m/s in the x direction and 0.2 in the y direction. ' + \
            'A value of 1.0 under "r0-r1 attention" means at that timestep, robot 0 is paying attention to robot 1, while a value of 0.0 means it is not. ' + \
            'A value of 3.4 under "r0-r1 distance" means that at that timestep, robot 0 and robot 1 are 3.4 meters apart. ' + \
            'Similar logic applies to all other pairings of robots. ' + \
            '\n\n' + \
            'Note that a collision has occurred between two robots when the distance between them is less than 0.05 m. ' + \
            'Please interpret the policy’s learned behavior based on the input table. Support your interpretation with data from the table. If you notice that the robots have collided (distance less than 0.05 m), please highlight this. \n\n'

    # TODO: add examples here
    # prompt += "Here are some examples of good interpretations:"
    tables = []
    interpretations = []
    for i in range(len(tables)):
        prompt += generate_table_example(tables[i], interpretations[i])
    
    return prompt

def build_prompt(new_table):
    # add instructions + few-shot learning examples
    prompt = build_preamble()

    # add actual data to analyze
    prompt += generate_table_example(new_table, "")
    return prompt


def main():
    # read input table
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)

    input_filename = sys.argv[1]

    with open(input_filename, 'r') as file:
        table = file.read()

    # fill in prompt
    prompt = build_prompt(table)
    print(prompt)

    # save to file
    with open(f'{input_filename}-prompt.txt', 'w') as f:
        f.write(prompt)


if __name__ == "__main__":
    main()
