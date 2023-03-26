import random
from copy import *
import numpy as np
import json

mapping = {0: 0, 1: 1, 2: -10, 3: -5, 4: 0, 5: 100}
encoded_map = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
               [2, 2, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
               [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
               [2, 0, 1, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0],
               [0, 2, 1, 0, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 1, 0, 2, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
               [2, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 4, 0, 0, 1, 0, 0, 3, 0, 0, 1, 1, 1, 1, 2, 2],
               [0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 3, 0],
               [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]



true_map = deepcopy(encoded_map)
for i in range(len(true_map)):
  for j in range(len(true_map[0])):
    true_map[i][j] = mapping[true_map[i][j]]

def reset_map_critical_val(original, new):
  for i in range(len(original)):
    for j in range(len(original[0])):
      if original[i][j] == 1:
        new[i][j] = "WALL"
      if original[i][j] == 2:
        new[i][j] = -10
      if original[i][j] == 3:
        new[i][j] = -5
      if original[i][j] == 5:
        new[i][j] = 200

reset_map_critical_val(encoded_map, true_map)

for i in true_map:
  print(i)
print()


# Hyperparams
DISCOUNT = 0.95
THETA = 0.01
P = 0.0

# State Params
NUM_ACTIONS = 4
ACTIONS = ["D", "L", "U", "R"]
ACTIONS_MAP = {"D": (1, 0), "L": (0, -1), "U": (-1, 0), "R": (0, 1)} # Down, Left, Up, Right
NUM_ROW = len(encoded_map)
NUM_COL = len(encoded_map[0])
NUM_STATES = NUM_ROW * NUM_COL
END_STATE = (2, 12)
MOVE_COST = -1

states = set()
rewards = {}
for r in range(NUM_ROW):
    for c in range(NUM_COL):
        if true_map[r][c] != "WALL":
            states.add((r, c))
            rewards[(r, c)] = true_map[r][c]

transition_probs = {}
for state in states:
    for action in ACTIONS:
        probs = []
        acts = deepcopy(ACTIONS)

        nx, ny = state[0]+ACTIONS_MAP[action][0], state[1]+ACTIONS_MAP[action][1]
        if nx < 0 or nx >= NUM_ROW or ny < 0 or ny >= NUM_COL or encoded_map[nx][ny] == 1:
            a1 = [1 - P, state, MOVE_COST, state == END_STATE]
        else:
            a1 = [1 - P, (nx, ny), MOVE_COST, (nx, ny) == END_STATE]

        probs.append(a1)
        acts.remove(action)

        for rem_acts in acts:
            nx, ny = state[0] + ACTIONS_MAP[rem_acts][0], state[1] + ACTIONS_MAP[rem_acts][1]
            if nx < 0 or nx >= NUM_ROW or ny < 0 or ny >= NUM_COL or encoded_map[nx][ny] == 1:
                a1 = [P / 3, state, MOVE_COST, state == END_STATE]
            else:
                a1 = [P / 3, (nx, ny), MOVE_COST, (nx, ny) == END_STATE]
            probs.append(a1)

        transition_probs[(state, action)] = probs



def printEnvironment(arr):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if encoded_map[r][c] == 1:
                val = "WALL"
            else:
                val = [".", "<-", "^", "->"][arr[r][c]]
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)




def policy_eval_fn(policy, graph):
    V = deepcopy(graph)

    # print("The Policy is")
    # for i in policy:
    #     print(i)

    # input()
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in states:
            v = 0
            # Look at the possible next actions
            a = policy[s[0]][s[1]]
                # For each action, look at the possible next states...

            for prob, next_state, reward, done in transition_probs[(s, ACTIONS[a])]:
                # Calculate the expected value
                # print(s, ACTIONS[a], prob, next_state, reward, done, V[next_state[0]][next_state[1]])
                # print("P({},{},{})".format(s, ACTIONS[a], next_state), " + ", "[r({},{},{}) + GAMMA * V({})]".format(s, ACTIONS[a], next_state, next_state))
                # print(prob, " + ", "[{} + {} * {}]".format(reward, DISCOUNT, V[next_state[0]][next_state[1]]))
                # print(prob * (0 + DISCOUNT * V[next_state[0]][next_state[1]]))
                # input()


                v += prob * (reward + DISCOUNT * V[next_state[0]][next_state[1]])

            # print("Cumulitive Val - ", v)
            # print()

            # How much our value function changed (across any states)
            if encoded_map[s[0]][s[1]] in [1, 5]:
                if encoded_map[s[0]][s[1]] == 1:
                    V[s[0]][s[1]] = "WALL"
                if encoded_map[s[0]][s[1]] == 5:
                    V[s[0]][s[1]] = 200
                delta = max(delta, 0)
            else:
                delta = max(delta, np.abs(v - V[s[0]][s[1]]))
                V[s[0]][s[1]] = round(v, 10)


        # for i in V:
        #     print(list(i))
        # print()
        # print(delta)
        # input()

        # Stop evaluating once our value function change is below a threshold
        if delta < THETA:
            # for i in V:
            #     print(list(i))
            # print()
            # input()
            break
    return V



def policy_improvement(init_policy, graph):

    def one_step_lookahead(state, V):
        A = [0, 0, 0, 0]
        for a in range(NUM_ACTIONS):
            # print(state, ACTIONS[a])
            for prob, next_state, reward, done in transition_probs[(state, ACTIONS[a])]:
                # print("P({},{},{})".format(s, ACTIONS[a], next_state), " * ", "[r({},{},{}) + GAMMA * V({})]".format(s, ACTIONS[a], next_state, next_state))
                # print(prob, " + ", "[{} + {} * {}]".format(reward, DISCOUNT, V[next_state[0]][next_state[1]]))
                # print(prob * (0 + DISCOUNT * V[next_state[0]][next_state[1]]))
                # input()
                A[a] += prob * (reward + DISCOUNT * V[next_state[0]][next_state[1]])
            # print(A[a])
        # print(np.argmax(A))
        # print(ACTIONS[np.argmax(A)])
        # input()
        return A

    policy = init_policy
    V = graph
    new_policy = np.zeros_like(policy)
    # print(new_policy)

    while True:
        V = policy_eval_fn(policy, V)
        stable = True

        for s in states:
            chosen_a = policy[s[0]][s[1]]

            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                stable = False

            new_policy[s] = best_a

        policy = new_policy.tolist()

        # for i in policy:
        #     print(list(i))
        # print()

        if stable:
            return policy, V




# Policy iteration
init_policy = [[1] * NUM_COL] * NUM_ROW


for i, num in enumerate(encoded_map):
    print(i, num)

print()

# for i in init_policy:
#     print(i)

policy, value_graph = policy_improvement(init_policy, true_map)

print("The policy is")
for i in policy:
    print(i)
print()

print("The value graph is")
for i in value_graph:
    print(i)
print()
# Print the optimal policy
print("The optimal policy is:\n")
printEnvironment(policy)


op_file = json.dumps([policy, value_graph])
file_name = "PolicyIteration - Discount-{}, THETA-{}, P-{}.json".format(DISCOUNT, THETA, P)

with open(file_name, "w") as outfile:
    json.dump(op_file, outfile, ensure_ascii=False)

outfile.close()