from copy import deepcopy
from snake import *
from GNN import *
import numpy as np

GNN_LAYER_SIZES = [24, 8, 4]

"""
Evaluate cardinals and diagonals (8 directions) 
and the distances from the head of various objects (food, wall, body) along those directions
"""
def get_inputs(state):
    view_rays = np.zeros(shape=(24, 1), dtype=np.float64)

    # 1st - wall - (0 - 7)
    # 2nd - food - (8 - 15)
    # 3rd - body - (16 - 23)

    # closer to 1, closer the obj.

    head = bin2idx(state.head)

    head_loc = (head // state.side_len, head % state.side_len) # row, col

    # check wall locations

    view_rays[0,0] = head_loc[0] # downward ray
    view_rays[1,0] = head_loc[1] # leftward ray
    view_rays[2,0] = state.side_len - view_rays[0,0] - 1 # upward ray
    view_rays[3,0] = state.side_len - view_rays[1,0] - 1 # rightward ray

    view_rays[4,0] = np.minimum(view_rays[0,0], view_rays[1,0]) # left/down-ward ray
    view_rays[5,0] = np.minimum(view_rays[1,0], view_rays[2,0]) # left/up-ward ray
    view_rays[6,0] = np.minimum(view_rays[2,0], view_rays[3,0]) # right/up-ward ray
    view_rays[7,0] = np.minimum(view_rays[3,0], view_rays[0,0]) # right/down-ward ray

    food = bin2idx(state.food)
    food_loc = (food // state.side_len, food % state.side_len) # row, col
    diff_row = head_loc[0] - food_loc[0]
    diff_col = head_loc[1] - food_loc[1]

    # check food location

    if diff_row == 0:
        if diff_col > 0:
            view_rays[8,0] = diff_col # downward ray
        else:
            view_rays[9,0] = -diff_col # leftward ray

    if diff_col == 0:
        if diff_row > 0:
            view_rays[10,0] = diff_row # upward ray
        else:
            view_rays[11,0] = -diff_row # rightward ray

    
    if diff_row == diff_col: # on a diagonal
        if diff_row < 0:
            view_rays[12,0] = abs(diff_row) # left/down-ward ray
        else:
            view_rays[13,0] = abs(diff_row) # left/up-ward ray

    if diff_row == -diff_col: # on a diagonal
        if diff_row < 0:
            view_rays[14,0] = abs(diff_row) # right/up-ward ray
        else:
            view_rays[15,0] = abs(diff_row) # right/down-ward ray

    # check body segment locations
    for body in state.body_list:
        b = bin2idx(body)
        loc = (b // state.side_len, b % state.side_len)

        diff_row = head_loc[0] - loc[0]
        diff_col = head_loc[1] - loc[1]

        if diff_row == 0:
            if diff_col < 0:
                if view_rays[16,0] == 0:
                    view_rays[16,0] = -diff_col # downward ray
                else:
                    view_rays[16,0] = min(view_rays[16,0], -diff_col) # downward ray

            else:
                if view_rays[17,0] == 0:
                    view_rays[17,0] = diff_col # leftward ray
                else:
                    view_rays[17,0] = min(view_rays[17,0], diff_col) # leftward ray

        if diff_col == 0:
            if diff_row < 0:
                if view_rays[18,0] == 0:
                    view_rays[18,0] = -diff_row # upward ray
                else:
                    view_rays[18,0] = min(view_rays[18,0], -diff_row) # upward ray

            else:
                if view_rays[19,0] == 0:
                    view_rays[19,0] = diff_row # rightward ray
                else:
                    view_rays[19,0] = min(view_rays[19,0], diff_row) # rightward ray

        if diff_row == diff_col: # on a diagonal
            if diff_row < 0:
                if view_rays[20,0] == 0:
                    view_rays[20,0] = abs(diff_row) # left/down-ward ray
                else:
                    view_rays[20,0] = min(view_rays[20,0], abs(diff_row)) # left/down-ward ray
            else:
                if view_rays[21,0] == 0:
                    view_rays[21,0] = abs(diff_row) # left/up-ward ray
                else:
                    view_rays[21,0] = min(view_rays[20,0], abs(diff_row)) # left/up-ward ray

        if diff_row == -diff_col: # on a diagonal
            if diff_row < 0:
                if view_rays[22,0] == 0:
                    view_rays[22,0] = abs(diff_row) # right/up-ward ray
                else:
                    view_rays[22,0] = min(view_rays[22,0],abs(diff_row)) # right/up-ward ray
            else:
                if view_rays[23,0] == 0:
                    view_rays[23,0] = abs(diff_row) # right/down-ward ray
                else:
                    view_rays[23,0] = min(view_rays[23,0],abs(diff_row)) # right/down-ward ray

    view_rays[view_rays != 0] = (state.side_len - view_rays[view_rays != 0] - 1) / (state.side_len - 2)

    return view_rays

def learn(num_brains, num_generations, display_best = False, step_food_ratio = 100):

    top_percentage = 0.1
    num_top_brains = int(num_brains * top_percentage) #top 10% stay alive in each gen
    num_elim_brains = num_brains - num_top_brains
    trial_brains = [GeneticNeuralNetwork(sizes=GNN_LAYER_SIZES, weight_clipping=False, mutation_chance=0.5, weight_deviation=1.0, bias_deviation=0.01) 
                    for _ in range(num_brains)]

    shared_state = Board()
    if display_best:
        screen = Display(frame_rate=60, userfocus=False, autoreset=False)

    for i in range(num_generations):
        fitness_scores = np.zeros(shape=num_brains)
        most_food_eaten = 0
        
        # gathering scores for each gnn
        for j in range(num_brains):
            shared_state.reset()
            while not shared_state.end and (shared_state.num_moves / (1. + shared_state.food_eaten)) < step_food_ratio:
                inputs = get_inputs(shared_state)
                brain_output = trial_brains[j].predict(inputs)

                move = np.argmax(brain_output)

                shared_state.push(move)
                
            fitness = 1000 * shared_state.food_eaten + shared_state.num_moves
            most_food_eaten = max(most_food_eaten, shared_state.food_eaten)
            fitness_scores[j] = fitness

        print(f"Generation: {i}")
        print(f"\tBest fitness: {np.max(fitness_scores)}")
        print(f"\tAvg fitness: {np.mean(fitness_scores)}")
        print(f"\tMost food eaten: {most_food_eaten}")

        max_index = np.argmax(fitness_scores)
        best_brain = deepcopy(trial_brains[max_index])

        if display_best:
            def brain_input(state):
                brain_output = best_brain.predict(get_inputs(state))

                move = np.argmax(brain_output)
                return move
            screen.show(input_func=brain_input, stop_loops=True, step_food_ratio=step_food_ratio)
            print(f"\tDisplayed snake's food eaten: {screen.state.food_eaten}")
            screen.state.reset()

        sorted_score_idxs = np.argsort(fitness_scores)

        for b in range(num_elim_brains):
            trial_brains[sorted_score_idxs[b]].load_info(trial_brains[sorted_score_idxs[num_elim_brains + (b % num_top_brains)]].copy_info())
        
        for b in trial_brains:
            b.mutate()
    screen.stop()
    best_brain.save(f'./SnakeGame/snake_brain_{time.time()}')

if __name__ == '__main__':
    population = 1000
    generations = 5000

    learn(num_brains=population, num_generations=generations, display_best=True, step_food_ratio=100)