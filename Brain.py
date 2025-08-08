import numpy as np
import concurrent.futures
import pygame
from JumpKing import JKGame
import time
import os
import datetime
import shutil


class Player():
    '''Player class

    A player represents one playthrough of the game, it consists
    of a list of inputs (actions), and has the utilities to "play" the game.

    In the sense of the genetic algorithm, the player's genetic material is
    represented by a list of uint8 values, each representing an action.

    An action consists of an action code and its duration.
    It can be coded as a uint8, where the 2 most important bits (first ones)
    are a binary code of the action code, defined by the agentCommand_dict,
    and the last 6 bits represent the duration in frames.

    agentCommand_dict = {
        0: 'right',
        1: 'left',
        2: 'right+space',
        3: 'left+space',
    }

    '''

    def __init__(self, action_count, actions_binary=None):
        self.action_count = action_count

        if actions_binary is not None:
            self.actions_binary = actions_binary
        else:
            self.actions_binary = np.random.randint(0, 256,
                size=self.action_count, dtype=np.uint8)

        # player's fitness function
        self.f = np.inf
        # current action has been active for this number of frames
        self.current_action_frames = 0
        # self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.actions = None
        self.current_action = 0
        self.no_more_actions = False
        # in-game time in seconds, used to calculate fitness
        self.time = 100
        self.completed_level = False

    def reset(self, action_count):
        '''
        Reset the player.
        '''
        self.action_count = action_count
        self.actions_binary = np.random.randint(0, 256, size=self.action_count, dtype=np.uint8)
        self.f = np.inf
        self.current_action_frames = 0
        self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        self.current_action = 0
        self.no_more_actions = False
        self.time = 100
        self.completed_level = False

    def load_from_save(filepath):
        '''Load a player from a save.

        Loads a player from a save created by ___

        Parameters
        ----------
        filepath : str
            The path on which the save is located.
        '''
        with open(filepath, 'r') as f:
            actions = np.empty(0, dtype=np.uint8)
            for line in f.readlines():
                if line[0] == str(0):
                    # actions = line.split(', ')
                    # actions = [x[2:] for x in actions]
                    # actions = [np.uint8(int(x, 2)) for x in actions]

                    binary_values = [s.strip() for s in line.split(',') if s.strip()]
                    actions = np.concatenate((actions, np.array([np.uint8(int(b, 2)) for b in binary_values])), dtype=np.uint8)

            player = Player(len(actions), actions)
            return player

    def print(self):
        '''Print the player's actions to the stdout

        '''
        if self.actions is None:
            self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        for action in self.actions:
            print(action)


    def get_agentCommand(self):
        '''Returns a command that this player would play, tracks time.

        This function should be called every frame of the game and it will return the
        player's action as a integer represeting the code of the action defined by the
        agentCommand_dict.

        The player will play every action from the self.actions list, as many times as
        the action's duration defines, after playing all available actions,
        self.no_more_actios is set to True, and this funciton will return 0.

        '''

        self.current_action_frames += 1
        if self.actions is None:
            self.actions = [self._bin_to_num(x) for x in self.actions_binary]
        if(self.current_action_frames > self.actions[self.current_action]["duration"]):
            self.current_action += 1
            self.current_action_frames = 0

        if self.current_action >= self.action_count:
            self.no_more_actions = True
            return 0

        action = self.actions[self.current_action]

        return action["agentCommand"]


    def _bin_to_num(self, binary_action : np.uint8):
        '''utility function to transform a binary action to an action dict
        '''
        agentCommand = int((binary_action & 0b11000000) >> 6)
        duration = np.uint8(binary_action & 0b00111111)

        action = {
            "agentCommand" : agentCommand,
            "duration"  : duration
        }
        return action

    def calculate_f(self, env, starting_state):
        '''calculates the fitness function

        Parameters
        ----------
        starting_state : dict
            dict represeting the starting state of the game.
            This enables creating checkpoints.

        state = {
            "level": 		0,
            "x": 			230,
            "y": 			298,
        }

        '''

        agentCommand_dict = {
            0: 'right',
            1: 'left',
            2: 'right+space',
            3: 'left+space',
        }


        state = env.reset_to_checkpoint(starting_state)
        # set the environment's fps to something large so the computation
        # happens as fast as possible
        env.fps = 10000

        # current level
        start_level = state["level"]

        if self.actions is None:
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

        self.time = fps_count / 60 # s, in-game time


        # # if the player completed a level, its fitness function will be just the time it took
        # if state["level"] > start_level:
        #     self.f = self.time
        #     self.completed_level = True
        #     print("juhu!")
        # else:
        #     # else f is the time + 100 * distance to the next level
        #     # trajanje nivoa u sekundama + 100 * koliko mu fali do vrha nivoa, y je izmedju 0 i 365 ili tako nesto
        #     # self.time je trajanje partije u sekundama, ocekujemo da ce biti 240s u minimumu
        #     #
        #     self.f = self.time + 100 * (state["y"] + 360 * (start_level - state["level"]))

        self.f = ( state["level"] * state["screen_height"] - state["y"] ) - self.time



    def show_replay(self, env, starting_state, fps):
        '''shows the replay of a player playing the game

        Parameters
        ----------
        env : JKGame
            Current game environment

        starting_state : dict
            dict represeting the starting state of the game.
            This enables creating checkpoints.

        state = {
            "level": 		0,
            "x": 			230,
            "y": 			298,
        }

        '''

        state = env.reset_to_checkpoint(starting_state)
        env.fps = fps
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
        '''Creates 4 offspring from 2 parents

        Randomly generates 2 indices where genetic materials of parents will be split
        and merged mixed, the genetic material is represented as a list of uint8 values
        each representing an action.

        A splitting index can be between 0 and 8 * k, where k is the number of actions
        a parent has.

        '''

        kids = []
        # calculate 2 indices where the parent's genetic material will be split
        splitting_index = np.random.randint(0, 8 * parent1.action_count, size=2)

        # for each index create 2 kids by mixing material
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
        '''mutate every bit of the player's genetic material with a chance of mutation_chance
        '''
        random = np.random.uniform(0, 1, self.action_count*8)
        for i in range(random.size):
            if random[i] < mutation_chance:
                j = i // 8
                x = i % 8
                self.actions_binary[j] = self.actions_binary[j] ^ (0b1 << x)



def player_calc_f(players, state):
    env = JKGame(wanna_blit=False)
    best_f = -np.inf
    best_player = None
    for player in players:
        player.calculate_f(env, state)

        if player.f > best_f:
            best_f = player.f
            best_player = player
    env.save_exit()
    return best_f, best_player

class Population():
    '''Population

    Class that represents the population in the genetic algorithm.
    A population consists of players.


    '''

    def __init__(self, size, action_count, mutation_chance, crossover_chance, max_gen, starting_state):
        self.size = size
        self.action_count = action_count
        self.players = [Player(self.action_count) for _ in range(size)]
        self.best_f = -np.inf
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
        self.best_f = -np.inf
        self.best_player = self.players[0]
        self.parents = []
        self.Nparents = self.size//5
        self.best_time = 100
        self.completed_level = False
        self.starting_state = starting_state





    def calculate_f(self):
        '''calculate fitness functions for every player in the generation
        '''

        self.batch_count = 5

        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(0, len(self.players), self.batch_count):
                max_b = min(i+self.batch_count, len(self.players))
                futures.append(executor.submit(player_calc_f, self.players[i:max_b], self.starting_state))

        for future in concurrent.futures.as_completed(futures):
            fitness, player = future.result()
            if fitness > self.best_f:
                self.best_f = fitness
                self.best_player = player



    def end(self):
        '''check if the algorithm ended
        '''
        # if self.best_f == 0:
        #     return True
        # else:
        #     return False
        return False

    def selection(self):
        '''perform selection

        From the population choose Nparents players using the binary
        tournament draft.

        Randomly choose 2 players and move the better of the two
        to the parents list, remove it from the player base.
        Until Nparents are chosen.

        '''
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
        '''perform crossover

        Randomly choose 2 parents and make 4 children, untill the whole
        new generation is generated.

        '''
        self.players = []

        # for i in range(0, self.Nparents-1, 2):
        while len(self.players) < self.size:
            parent1 = self.parents[np.random.randint(0, len(self.parents))]
            parent2 = self.parents[np.random.randint(0, len(self.parents))]

            if np.random.random() < self.crossover_chance:
                kids = Player.create_kids(parent1, parent2)

                spots = self.size - len(self.players)
                if spots < 4:
                    self.players = self.players + kids[0:spots]
                else:
                    self.players = self.players + kids


    def mutation(self):
        '''mutate every player in the generation'''
        for player in self.players:
            player.mutate(self.mutation_chance)

    def optimize(self):
        '''main optimization loop'''

        gen = 1
        start_time = time.time()

        while gen < self.max_gen:
            self.calculate_f()
            self.selection()
            self.crossover()
            self.mutation()
            print(f'generation: {gen}, best_solution: {self.best_f}, time_elapsed: {(time.time() - start_time)/60:.2f}min')
            print(f'completed_level: {self.best_player.completed_level}, best_time: {self.best_player.time:.2f}s')
            gen +=1


        # make a save
        date = datetime.datetime.now()
        filepath = os.path.join('Saves', f"{date:%d-%m-%y-%H-%M-%S}.txt")

        actions: np.typing.NDArray[np.uint8] = self.best_player.actions_binary

        with open(filepath, "w+") as f:
            f.write('solution found at: ' + date.strftime("%c") + '\n')
            f.write(f'generation: {gen}, time_elapsed: {(time.time() - start_time)/60:.2f}min ')
            f.write(f'completed_level: {self.best_player.completed_level}, in game time: {self.best_player.time}s\n')
            f.write(f'player fitness level: {self.best_player.f}\n')
            f.write(f'population size: {self.size}, action count: {self.action_count}, mutation_chance: {self.mutation_chance} ')
            f.write(f'crossover_chance: {self.crossover_chance}\n')

            for action in actions:
                f.write(bin(action))
                f.write(', ')
            f.write('\n')

        with open(os.path.join("Saves", "latest.txt"), "a+") as f:
            for action in actions:
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


    player = Player.load_from_save(str(os.path.join('Saves', 'latest.txt')))

    env = JKGame()

    end_state = player.show_replay(env, state, 10000)

    pop = Population(
        size=50,
        action_count=7,
        mutation_chance=0.15,
        crossover_chance=0.8,
        max_gen=10,
        starting_state=end_state
    )

    player = pop.optimize()
    env.save_exit()

    # pop.reset(
    #     size=50,
    #     action_count=8,
    #     starting_state=end_state
    # )

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
    # player = Player.load_from_save("Saves\\25-05-25-17-11-38.txt")
    # player.print()
    # player.show_replay(pop.env, end_state)

    print('end')
