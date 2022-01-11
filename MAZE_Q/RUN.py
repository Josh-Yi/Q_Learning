from maze_env import Maze
from Q_Brain import QLearningTable

def update():
    for episode in range (5):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_


            if done:
                break


    print('GAME OVER')
    env.destory()


if __name__ =='__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()