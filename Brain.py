import numpy as np
import math
from JumpKing import JKGame
from itertools import chain
import time
import os

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
            # self.actions_binary = [ 
            #     0b00111111, 0b01111111, 0b10111111,
            #     0b11111111, 0b10111111
            # ]
            

        self.f = math.inf

        self.current_action_frames = 0
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.current_action = 0
        self.no_more_actions = False

    
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

    def calculate_f(self, env):
        agentCommand_dict = {
            0: 'right',
            1: 'left',
            2: 'right+space',
            3: 'left+space',
        }

        state = env.reset()
        env.fps = 1000
        start_level = state["level"]

        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.no_more_actions = False
        self.current_action = 0

        last_state = state

        while not (self.no_more_actions and state["move_available"]):
            # if state["level"] > start_level:
            #     env.fps = 30

            if state["move_available"]:
                agentCommand = self.get_agentCommand()
                # print(agentCommand_dict[agentCommand])
                state = env.step(agentCommand)
            else:
                state = env.step(0)

        
        if state["level"] > start_level:
            self.f = 0
            print("juhu!")
        else:
            self.f = state["y"]
    
    def show_replay(self, env):
        state = env.reset()
        env.fps = 60
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.no_more_actions = False
        self.current_action = 0

        while not self.no_more_actions:

            if state["move_available"]:
                agentCommand = self.get_agentCommand()
                state = env.step(agentCommand)
            else:
                state = env.step(0)
            
        

        
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
    def __init__(self, size, action_count, mutation_chance):
        self.size = size
        self.action_count = action_count
        self.players = [Player(self.action_count) for _ in range(size)]
        self.env = JKGame()
        self.best_f = math.inf
        self.best_player = self.players[0]
        self.parents = []
        self.Nparents = self.size//2
        self.mutation_chance = mutation_chance


    def quit_env(self):
        self.env.save_exit()

    def calculate_f(self):
        for player in self.players:
            self.env.reset()
            player.calculate_f(self.env)
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

        for i in range(0, self.Nparents-1, 2):

            parent1 = self.parents[i]
            parent2 = self.parents[i+1]
            
            kids = Player.create_kids(parent1, parent2)
            self.players = list(chain(self.players, kids))


    def mutate(self):
        for player in self.players:
            player.mutate(self.mutation_chance)
    
    def optimize(self):
        gen = 1
        start_time = time.time()

        while True:
            self.calculate_f()

            if(self.end()):
                print("found best solution")
                print(f'generation: {gen}, best_solution: {self.best_f}, time_elapsed: {time.time() - start_time}')
                print(self.best_player.actions)
                break


            self.selection()
            self.crossover()
            self.mutate()
            print(f'generation: {gen}, best_solution: {self.best_f}, time_elapsed: {time.time() - start_time}')
            gen +=1
        
        
        return self.best_player



    
    
if __name__ == "__main__":
    # pop = Population(10, 5, 0.1)
    pop = Population(50, 4, 0.1)

    player = pop.optimize()
    time.sleep(2)
    player.show_replay(pop.env)
    pop.quit_env()

    print('end')
