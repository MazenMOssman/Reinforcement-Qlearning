from agent import Agent
if __name__ == "__main__":
    Test = Agent()
    Q,policy,V = Test.play(alpha=1, gamma=0.99, epsilon=1,time_step=50, episode=100)
    Test.view(policy,stat = 5)
    Test.Value_state(V)
