import sys


def read_input_arms():

    # TODO: assert if (len(sys.argv) != 3): raise Exception()

    actions = {}
    with open(sys.argv[1], 'r') as arms:
        for line in arms:
            features = line.strip().split(" ")
            id = int(features[0])
            actions[id] = [float(x) for x in features[1:]]
    return actions



def get_data(self, line):
    log_line = line.strip().split()
    time = int(log_line[0])
    chosen = int(log_line.pop(7))
    reward = int(log_line.pop(7))
    user_features = [float(x) for x in log_line[1:7]]
    articles = [int(x) for x in log_line[7:]]

    return((time, articles, user_features, chosen, reward))

