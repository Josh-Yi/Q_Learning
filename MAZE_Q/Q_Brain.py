import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self,actions, lr=0.01, reward_decay=0.9,greedy=0.9):
        self.actions = actions
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon = greedy
        self.q_table = pd.DataFrame(columns=self.actions)


    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.uniform(0,1)<self.epsilon:
            # state_action = self.q_table.iloc[observation][:]
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self,state, action, reward, state_):
        self.check_state_exist(state_)
        q_predict = self.q_table.loc[state,action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state,:].max()
        else:
            q_target = reward

        self.q_table.loc[state,action] += self.lr * (q_target - q_predict)
        print(self.q_table)

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name = state
                )
            )

