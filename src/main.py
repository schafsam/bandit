import time
from src.policy import Recommender
from src.utility import *

__author__ = "Samuel Schaffhauser"

# in terms of the paper this is the evaluator
# arms are equivalent to actions
# the program is executed in this fashion:
#   python main.py arms.txt content.txt

# TODO: include in the recommender class the score and number_updates!


if __name__ == "__main__":

    start = time.clock()
    recommender = Recommender()
    arms  = read_input_arms()
    recommender.set_arms(arms)

    score = 0
    seen_lines = 0
    number_updates = 0


    with open(sys.argv[2], 'r') as inf:
        for line in inf:
            seen_lines += 1

            time, arm, context, chosen, reward = get_data(line=line)

            calculated = recommender.recommend(time=time, articles=arm, user_features=context)

            if calculated == chosen:
                recommender.update(reward)
                score += reward
                number_updates += 1
            else:
                recommender.update(-1)

        print("Evaluated ", number_updates / seen_lines)
        print("CTR= ",float(score) / number_updates)
        print("Time expected: ", float((time.clock()-start)*5000/60))
