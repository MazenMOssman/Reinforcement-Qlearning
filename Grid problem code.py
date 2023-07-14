import numpy as np

class Env():
    def __init__(self):
        #Currentstate:{action:[probability,next state,reward,terminal or no terminal]}
        self.Grid = {0:{0:[1,0,-1,False],
                  1:[1,1,-1,False],
                  2:[1,5,-1,False],
                  3:[1,0,-1,False]},

                    1:{0:[1,1,-1,False],
                  1:[1,2,-1,False],
                  2:[1,6,-1,False],
                  3:[1,0,-1,False]},


                    2:{0:[1,2,-1,False],
                  1:[1,3,-1,False],
                  2:[1,7,-1,False],
                  3:[1,1,-1,False]},

                    3:{0:[1,3,-1,False],
                  1:[1,4,-1,False],
                  2:[1,8,-1,False],
                  3:[1,2,-1,False]},

                    4:{0:[1,4,-1,False],
                  1:[1,4,-1,False],
                  2:[1,9,-1,False],
                  3:[1,3,-1,False]},

                    5:{0:[1,0,-1,False],
                  1:[1,6,-1,False],
                  2:[1,10,-1,False],
                  3:[1,5,-1,False]},

                    6:{0:[1,1,-1,False],
                  1:[1,7,-1,False],
                  2:[1,11,-1,False],
                  3:[1,5,-1,False]},

                    7:{0:[1,2,-1,False],
                  1:[1,8,-1,False],
                  2:[1,7,-1,False],
                  3:[1,6,-1,False]},

                    8:{0:[1,3,-1,False],
                  1:[1,9,-1,False],
                  2:[1,18,+5,False],
                  3:[1,7,-1,False]},

                    9:{0:[1,4,-1,False],
                  1:[1,9,-1,False],
                  2:[1,9,-1,False],
                  3:[1,8,-1,False]},

                    10:{0:[1,5,-1,False],
                  1:[1,11,-1,False],
                  2:[1,15,-1,False],
                  3:[1,10,-1,False]},

                    11:{0:[1,6,-1,False],
                  1:[1,11,-1,False],
                  2:[1,16,-1,False],
                  3:[1,10,-1,False]},

                    15:{0:[1,10,-1,False],
                  1:[1,16,-1,False],
                  2:[1,20,-1,False],
                  3:[1,15,-1,False]},

                    16:{0:[1,11,-1,False],
                  1:[1,16,-1,False],
                  2:[1,21,-1,False],
                  3:[1,15,-1,False]},


                    18:{0:[1,18,-1,False],
                  1:[1,19,-1,False],
                  2:[1,23,-1,False],
                  3:[1,18,-1,False],},


                    19:{0:[1,19,-1,False],
                  1:[1,19,-1,False],
                  2:[1,24,10,True],
                  3:[1,18,-1,False]},


                    20:{0:[1,15,-1,False],
                  1:[1,21,-1,False],
                  2:[1,20,-1,False],
                  3:[1,20,-1,False]},



                    21:{0:[1,16,-1,False],
                  1:[1,22,-1,False],
                  2:[1,21,-1,False],
                  3:[1,20,-1,False]},


                    22:{0:[1,22,-1,False],
                  1:[1,23,-1,False],
                  2:[1,22,-1,False],
                  3:[1,21,-1,False]},


                    23:{0:[1,18,-1,False],
                  1:[1,24, 10,True],
                  2:[1,23,-1,False],
                  3:[1,22,-1,False]},

                    24:{0:[1,24,0,True],
                  1:[1,24,0,True],
                  2:[1,24,0,True],
                  3:[1,24,0,True]},

                    12: {0: [1, 12, 0, False],
                         1: [1, 12, 0, False],
                         2: [1, 12, 0, False],
                         3: [1, 12, 0, False]},
                    13: {0: [1, 13, 0, False],
                         1: [1, 13, 0, False],
                         2: [1, 13, 0, False],
                         3: [1, 13, 0, False]},
                    14: {0: [1, 14, 0, False],
                         1: [1, 14, 0, False],
                         2: [1, 14, 0, False],
                         3: [1, 14, 0, False]},
                    17: {0: [1, 17, 0, False],
                         1: [1, 17, 0, False],
                         2: [1, 17, 0, False],
                         3: [1, 17, 0, False]}}






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


if __name__ == "__main__":
    Test = Agent()
    Q,policy,V = Test.play(alpha=1, gamma=0.99, epsilon=1,time_step=50, episode=100)
    Test.view(policy,stat = 5)
    Test.Value_state(V)
