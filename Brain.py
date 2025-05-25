import numpy as np
import math
from JumpKing import JKGame
from itertools import chain
import time
import os
import pygame
import datetime

class Brain():

    action_dict: dict

    def __init__(self):
        self.population = Population()

    def optimize(self):

        while True:
            Population.calculate_f()

            if(Population.end()):
                break

            Population.select_parents()
            Population.crossover()
            Population.mutate()

class Player():
    
    def __init__(self, action_count, actions_binary=None):
        self.action_count = action_count
        if actions_binary is not None:
            self.actions_binary = actions_binary
        else:
            self.actions_binary = np.random.randint(0, 256, size=self.action_count, dtype=np.uint8)
            
        self.f = math.inf
        self.current_action_frames = 0
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.current_action = 0
        self.no_more_actions = False
        self.time = 100
        self.completed_level = False
    
    def reset(self, action_count):
        self.action_count = action_count
        self.actions_binary = np.random.randint(0, 256, size=self.action_count, dtype=np.uint8)
        self.f = math.inf
        self.current_action_frames = 0
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.current_action = 0
        self.no_more_actions = False
        self.time = 100
        self.completed_level = False

    def load_from_save(filepath):
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line[0] == str(0):
                    # actions = line.split(', ')
                    # actions = [x[2:] for x in actions]
                    # actions = [np.uint8(int(x, 2)) for x in actions]

                    binary_values = [s.strip() for s in line.split(',') if s.strip()]
                    actions = np.array([np.uint8(int(b, 2)) for b in binary_values], dtype=np.uint8)
        
        player = Player(len(actions), actions)
        return player
                

    
    def print(self):
        if self.actions is None:
            self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        for action in self.actions:
            print(action)

    
    def get_agentCommand(self):
        self.current_action_frames += 1
        if(self.current_action_frames > self.actions[self.current_action]["duration"]):
            self.current_action += 1
            self.current_action_frames = 0

        if self.current_action >= self.action_count:
            self.no_more_actions = True
            return 0

        action = self.actions[self.current_action]

        return action["agentCommand"]

    
    def _bin_to_num(self, binary_action : np.uint8):
        agentCommand = int((binary_action & 0b11000000) >> 6)
        duration = np.uint8(binary_action & 0b00111111)

        action = {
            "agentCommand" : agentCommand,
            "duration"  : duration
        }
        return action

    def calculate_f(self, env, starting_state):
        agentCommand_dict = {
            0: 'right',
            1: 'left',
            2: 'right+space',
            3: 'left+space',
        }

        # state = env.reset()
        state = env.reset_to_checkpoint(starting_state)
        env.fps = 10000
        start_level = state["level"]

        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.no_more_actions = False
        self.current_action = 0

        last_state = state
        fps_count = 0

        while not (self.no_more_actions and state["move_available"]):

            if state["move_available"]:
                agentCommand = self.get_agentCommand()
                # print(agentCommand_dict[agentCommand])
                state = env.step(agentCommand)
            else:
                state = env.step(0)

            fps_count += 1
        
        self.time = fps_count / 60 # s

        
        if state["level"] > start_level:
            self.f = self.time 
            self.completed_level = True
            print("juhu!")
        else:
            # trajanje nivoa u sekundama + 100 * koliko mu fali do vrha nivoa, y je izmedju 0 i 365 ili tako nesto
            self.f = self.time + 100 * (state["y"] + 360 * (start_level - state["level"]))
            
    
    def show_replay(self, env, starting_state):
        # state = env.reset()
        state = env.reset_to_checkpoint(starting_state)
        env.fps = 60
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.no_more_actions = False
        self.current_action = 0

        while not (self.no_more_actions and state["move_available"]):

            if state["move_available"]:
                agentCommand = self.get_agentCommand()
                state = env.step(agentCommand)
            else:
                state = env.step(0)
        
        return state
            
        

        
    def create_kids(parent1, parent2):
        kids = []
        splitting_index = np.random.randint(0, 8 * parent1.action_count, size=2)

        for i in splitting_index:
            if i % 8 == 0:
                j = int(i/8)
                actions_binary1 = np.concatenate((parent1.actions_binary[:j], parent2.actions_binary[j:]))
                # print(f"case 1 length: {len(actions_binary1)}")
                actions_binary2 = np.concatenate((parent2.actions_binary[:j], parent1.actions_binary[j:]))
            else:
                j = int(i/8)
                x = i%8

                chopped_huz1 = np.array([( parent1.actions_binary[j] & (0b1 << x) ) | ( parent2.actions_binary[j] & ~(0b1<<x) )], dtype=np.uint8)
                actions_binary1 = np.concatenate((parent1.actions_binary[0:j], chopped_huz1, parent2.actions_binary[j+1:]))

                chopped_huz2 = np.array([( parent2.actions_binary[j] & (0b1 << x) ) | ( parent1.actions_binary[j] & ~(0b1<<x) )], dtype=np.uint8)
                actions_binary2 = np.concatenate((parent2.actions_binary[0:j], chopped_huz2, parent1.actions_binary[j+1:]))
                # print(f"case2 length1: {len(actions_binary1)} length2: {len(actions_binary2)}")
            
            kid1 = Player(parent1.action_count, actions_binary1)
            kid2 = Player(parent1.action_count, actions_binary2)
            kids.append(kid1)
            kids.append(kid2)


        return kids
    
    def mutate(self, mutation_chance):
        random = np.random.uniform(0, 1, self.action_count*8)
        for i in range(random.size):
            if random[i] < mutation_chance:
                j = i // 8
                x = i % 8
                self.actions_binary[j] = self.actions_binary[j] ^ (0b1 << x)
    

        
class Population():
    def __init__(self, size, action_count, mutation_chance, crossover_chance, max_gen, starting_state):
        self.size = size
        self.action_count = action_count
        self.players = [Player(self.action_count) for _ in range(size)]
        self.env = JKGame()
        self.best_f = math.inf
        self.best_player = self.players[0]
        self.parents = []
        self.Nparents = self.size//5
        self.mutation_chance = mutation_chance
        self.max_gen = max_gen
        self.best_time = 100
        self.completed_level = False
        self.crossover_chance = crossover_chance
        self.starting_state = starting_state

    def reset(self, size, action_count, starting_state):
        self.action_count = action_count
        self.size = size
        for player in self.players:
            player.reset(action_count)
        self.best_f = math.inf
        self.best_player = self.players[0]
        self.parents = []
        self.Nparents = self.size//5
        self.best_time = 100
        self.completed_level = False
        self.starting_state = starting_state



    def quit_env(self):
        self.env.save_exit()

    def calculate_f(self):
        for player in self.players:
            self.env.reset()
            player.calculate_f(self.env, self.starting_state)
            if player.f < self.best_f:
                self.best_player = player
                self.best_f = player.f

    def end(self):
        if self.best_f == 0:
            return True
        else:
            return False

    def selection(self):
        self.parents = []

        for _ in range(self.Nparents):

            ind1 = np.random.randint(0, len(self.players))
            ind2 = np.random.randint(0, len(self.players))
            
            if self.players[ind1].f < self.players[ind2].f:
                self.parents.append(self.players[ind1])
                self.players.pop(ind1)
            else:
                self.parents.append(self.players[ind2])
                self.players.pop(ind2)
        
    def crossover(self):
        self.players = []

        # for i in range(0, self.Nparents-1, 2):
        while len(self.players) < self.size:
            parent1 = self.parents[np.random.randint(0, len(self.parents))]
            parent2 = self.parents[np.random.randint(0, len(self.parents))]
            
            if np.random.random() < self.crossover_chance:
                kids = Player.create_kids(parent1, parent2)

                self.players.append(kids[0])

                if self.size - len(self.players) != 1:
                    self.players.append(kids[1])


    def mutate(self):
        for player in self.players:
            player.mutate(self.mutation_chance)
    
    def optimize(self):
        gen = 1
        start_time = time.time()

        while gen < self.max_gen:
            self.calculate_f()
            self.selection()
            self.crossover()
            self.mutate()
            print(f'generation: {gen}, best_solution: {self.best_f}, time_elapsed: {(time.time() - start_time)/60:.2f}min')
            print(f'completed_level: {self.best_player.completed_level}, best_time: {self.best_player.time:.2f}s')
            gen +=1
        

        date = datetime.datetime.now()
        filepath = os.path.join('Saves', f"{date:%d-%m-%y-%H-%M-%S}.txt")

        with open(filepath, "w+") as f:
            f.write('solution found at: ' + date.strftime("%c") + '\n')
            f.write(f'generation: {gen}, time_elapsed: {(time.time() - start_time)/60:.2f}min ')
            f.write(f'completed_level: {self.best_player.completed_level}, in game time: {self.best_player.time}s\n')
            f.write(f'population size: {self.size}, action count: {self.action_count}, mutation_chance: {self.mutation_chance} ')
            f.write(f'crossover_chance: {self.crossover_chance}\n')

            for action in self.best_player.actions_binary:
                f.write(bin(action))
                f.write(', ')
            f.write('\n')

        return self.best_player
        
    
    
if __name__ == "__main__":

    state = {
        "level": 		0,
        "x": 			230,
        "y": 			298,
        }

    pop = Population(
        size=100,
        action_count=7,
        mutation_chance=0.15,
        crossover_chance=0.8,
        max_gen=50,
        starting_state=state
    )

    # player = pop.optimize()
    player = Player.load_from_save("Saves\\23-05-25-20-55-47.txt")

    end_state = player.show_replay(pop.env, state)

    pop.reset(
        size=50,
        action_count=8,
        starting_state=end_state
    )

    # player2 = pop.optimize()


    # players = []
    # for i in range(30):
    #     player = pop.optimize()
    #     players.append(player)

    #     end_state = player.show_replay(pop.env, state)

    #     pop.reset(
    #         size=50,
    #         action_count=6,
    #         starting_state=end_state
    #     )
        


    # player = Player.load_from_save(os.path.join("Saves","23-05-25-20-20-48.txt"))
    # player = Player.load_from_save("Saves\\23-05-25-20-55-47.txt")
    player = Player.load_from_save("Saves\\25-05-25-17-11-38.txt")
    # player.print()
    player.show_replay(pop.env, end_state)

    print('end')
