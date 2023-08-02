import random
import numpy as np
from binance import Client
import pandas as pd
import neat
import pickle
import os


# client = Client()
"""
global frame
frame = pd.read_csv("data.txt")
frame = pd.DataFrame(frame)
frame.columns = ['C']
frame = frame.astype(float)
"""
frame = np.load("frame.npy")

DATA_RANGE = 30
END = len(frame)
PLAY_TIME = 12*60
START_USDT = 1000.0
NUM_OF_GEN = 5


class AI:
    def __init__(self, start):
        self.BTC_ACC = 0.0
        self.starting_val = start
        self.USDT_ACC = start
        self.score = 0
        self.last = 0

    def buy(self, value):
        self.BTC_ACC += self.USDT_ACC/value
        self.score = self.USDT_ACC/self.starting_val
        self.USDT_ACC = 0.0

    def sell(self, value):
        self.USDT_ACC += self.BTC_ACC*value
        self.score = self.USDT_ACC/self.starting_val
        self.BTC_ACC = 0.0

"""
#symulation
def test(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    f = open("best_changed.pickle", 'rb')
    genome = pickle.load(f)
    nets = neat.nn.FeedForwardNetwork.create(genome, config)
    f.close()


    ai = AI(START_USDT)
    test_data = []
    f.close()

    for j in range(DATA_RANGE):
        test_data.append(float(frame['C'][j]))

    for i in range(PLAY_TIME - DATA_RANGE):

        output = nets.activate(test_data)
        k = 0.0
        decision = 0
        if output[0] > k:
            decision = 1
        if output[1] > k:
            decision = -1

        if decision == 1:
            ai.buy(float(frame['C'][DATA_RANGE + i]))
        elif decision == -1:
            ai.sell(float(frame['C'][DATA_RANGE + i]))
        del test_data[0]
        test_data.append(float(frame['C'][DATA_RANGE + i]))

    # scoring and fitness
    if ai.BTC_ACC > 0:
        ai.score = ai.BTC_ACC * float(frame['C'][PLAY_TIME - 1]) / START_USDT
    else:
        ai.score = ai.USDT_ACC / START_USDT

    print(ai.score)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    test(config_path)

"""
# TRADING
def play(genomes, config):
    max_score = 0
    nets = []
    ge = []
    ais = []
    test_data = []

    # creating ais
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        ais.append(AI(START_USDT))
        g.fitness = 0
        ge.append(g)

    day = 10
    for j in range(DATA_RANGE):
        test_data.append(float(frame[day][j]))

    for i in range(PLAY_TIME - DATA_RANGE-1):
        change = float(frame[day][DATA_RANGE + i + 1]) - float(frame[day][DATA_RANGE + i])
        # real trading
        for x, ai in enumerate(ais):
            output = nets[x].activate(test_data)
        
            k = 0.5
            decision = 0
            if output[0] > k:
                decision = 1
            if output[1] > k:
                decision = -1

            if decision == 1:
                ai.buy(float(frame[day][DATA_RANGE + i]))
                if change > 0:
                    ge[x].fitness += 1
                else:
                    ge[x].fitness -= 1
            elif decision == -1:
                ai.sell(float(frame[day][DATA_RANGE + i]))
                if change < 0:
                    ge[x].fitness += 1
                else:
                    ge[x].fitness -= 1
        del test_data[0]
        test_data.append(float(frame[day][DATA_RANGE + i]))

    # scoring and fitness
    for x, ai in enumerate(ais):
        if ai.BTC_ACC > 0:
            ai.score = ai.BTC_ACC * float(frame[day][PLAY_TIME - 1]) / START_USDT
        else:
            ai.score = ai.USDT_ACC / START_USDT
        if ai.score > max_score:
            max_score = ai.score
        if ai.score <= 1.00:
            ge[x].fitness -= 10 + int(100*(1.01 - ai.score))
        else:
            ge[x].fitness += int(200*(ai.score - 1.0))
            if ai.score > 1.04:
                ge[x].fitness += 100

    print("MAX = ", max_score, "DAY = ", day + 1)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to  num_of_gen generations.
    winner = p.run(play, NUM_OF_GEN)
    f = open("AI.pickle", "wb")
    pickle.dump(winner, f)
    f.close()

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

