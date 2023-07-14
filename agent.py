from Env import Env
import numpy as np
class Agent(Env):
    def __init__(self):
        self.start = 5
        self.actions = [0, 1, 2, 3]
        self.environment = Env().Grid

    def exp_expl(self, state,epsilon,Q_values):
        if np.random.rand() > epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(Q_values[state])
    def step(self,action,state):
        next_state = self.environment[state][action][1]
        reward = self.environment[state][action][2]
        done = self.environment[state][action][3]
        return next_state,reward,done

    def play(self, alpha=1, gamma=0.99, epsilon=1,time_step=50, episode=100):
        #best Epsilon for every state in the grid is 0.1
        Q_values = np.zeros((len(self.environment), len(self.environment[self.start])), dtype=np.float64)
        total =[]
        for ep in range(episode):
           ep_reward = 0
           state= self.start
           for iteration in range(time_step):
                   action = self.exp_expl(state, epsilon, Q_values)
                   next_state, reward, done = self.step(action, state)
                   Q_next = np.max(Q_values[next_state])
                   Q_values[state, action] += alpha * (reward + gamma * Q_next - Q_values[state,action])
                   ep_reward+=reward

                   state = next_state
           total.append(ep_reward)
        print("Cumulative reward =",np.mean(total))
        print(" ")
        print("NOTE TO GET CUMULATIVE REWARD = 10 THE EPISODE NUMBER SHOULD BE 150 ")
        print(" ")
        print("NOTE THE MODEL IS TRAINED ON EPSILON = 1")
        print(" ")
        policy = np.argmax(Q_values,axis =1)
        V = np.max(Q_values,axis = 1)
        return Q_values,policy,V







    def view(self,pi,stat = 5):
        state = stat
        done = False
        directions  = {0:"UP",
                       1:"RIGHT",
                       2:"DOWN",
                       3:"LEFT"}
        rew = 0
        while done == False:
            done = self.environment[state][pi[state]][3]
            print("State",state)
            print("Action",directions[pi[state]])
            next_state = self.environment[state][pi[state]][1]
            rew += self.environment[state][pi[state]][2]
            state = next_state
            print("Total Reward = ",rew)
            # time.time(2)

        return rew
    def Value_state(self,V):
        l = np.zeros([5,5])
        k = 0
        for i in range(5):
            for j in range(5):
                l[i][j] = V[k]
                k+=1
        print("\n")
        print("The state values  \n",l)

