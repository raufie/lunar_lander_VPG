from model import MLP
import torch
from torch.distributions import Categorical
import gym
def run_demo(n_iters = 1):

    device="cpu"

    agent = MLP(8,4, hidden_units=10)
    agent.load_state_dict(torch.load("./vanilla_pg_2.pt", map_location=device))
    
    # env
    env = gym.make("LunarLander-v2", render_mode="human")
    for _ in range(n_iters):
        done = False
        truncated = False
        s = env.reset()[0]
        while not done:
            
            s = torch.as_tensor(s).to(device)

            action = Categorical(logits = agent(s)).sample()
            s, r, done, truncated, _ = env.step(action.item())
        
            if truncated:
                break
    
if __name__ == "__main__":
    run_demo(n_iters=5)