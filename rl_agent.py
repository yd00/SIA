import keras.models
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.optimizers import Adam


class RLAgent:
    def build_agent(model, actions, NUMBER_OF_STEPS, MEMORY_LIMIT, ENABLE_DUELING_NETWORK, WARMUP_STEPS, ENABLE_DOUBLE_DQN):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_min=0.1, value_max=0.1, value_test=0.2, nb_steps=NUMBER_OF_STEPS)
        memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=3)
        agent = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=ENABLE_DUELING_NETWORK, dueling_type='avg', nb_actions=actions, nb_steps_warmup=WARMUP_STEPS, enable_double_dqn=ENABLE_DOUBLE_DQN)
        agent.compile(Adam(learning_rate=1e-4), metrics=['mae'])
        return agent

    def train_agent(agent, env, NUMBER_OF_STEPS):
        agent.fit(env, nb_steps=NUMBER_OF_STEPS, visualize=False, verbose=2)
        return agent
