import gym

from ale_py.roms import SpaceInvaders
from ale_py import ALEInterface
from dl_model import DLModel
from rl_agent import RLAgent

ale = ALEInterface()
ale.loadROM(SpaceInvaders)
env = gym.make('ALE/Pacman-v5', render_mode='human')
height, width, channels = env.observation_space.shape

EPISODES_TRAIN = 5
EPISODES_TEST = 5
WARMUP_STEPS = 1000
MEMORY_LIMIT = 1000

# SET THESE PARAMETERS TO LOAD APPROPRIATE WEIGHTS
NUMBER_OF_STEPS = 2000  #100, 200, 500, 1000, 2000, 5000
ENABLE_DUELING_NETWORK = True
ENABLE_DOUBLE_DQN = False
OPTIMISER = 'adadelta'  #adam, adadelta

actions = env.action_space.n
env.unwrapped.get_action_meanings()

model = DLModel.build_model(height, width, channels, actions)
dqn_agent = RLAgent.build_agent(model, actions, NUMBER_OF_STEPS, MEMORY_LIMIT, ENABLE_DUELING_NETWORK, WARMUP_STEPS, ENABLE_DOUBLE_DQN)


print(f'1. Load a model\n2. Create a new model')
option = int(input())

if(option == 1):
    dqn_agent.load_weights(f'saved_models/{OPTIMISER}/{"dueling" if ENABLE_DUELING_NETWORK else "no_dueling"}/{"ddqn" if ENABLE_DOUBLE_DQN else "dqn"}_{NUMBER_OF_STEPS}/{"ddqn" if ENABLE_DOUBLE_DQN else "dqn"}_{NUMBER_OF_STEPS}.h5f')
else:
    dqn_agent = RLAgent.train_agent(dqn_agent, env, NUMBER_OF_STEPS)

scores = dqn_agent.test(env, nb_episodes=EPISODES_TEST, visualize=False)


if(option != 1):
    print('Press 1 to save the weights. Press any other number to skip saving: ')
    save_model = int(input())
    #print(scores.history)
    if save_model == 1:
        model_path = f'saved_models/{OPTIMISER}/{"dueling" if ENABLE_DUELING_NETWORK else "no_dueling"}/{"ddqn" if ENABLE_DOUBLE_DQN else "dqn"}_{NUMBER_OF_STEPS}/{"ddqn" if ENABLE_DOUBLE_DQN else "dqn"}_{NUMBER_OF_STEPS}.h5f'
        dqn_agent.save_weights(model_path)
        print(f'Weights saved at: {model_path}')
