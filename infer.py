from sim.env import RANEnvironment
from utils.preprocess import preprocess
from models.model import SchedulerNet

import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SchedulerNet().to(device)
    model.load_state_dict(torch.load("models/scheduler.pth"))
    model.eval()

    env = RANEnvironment(num_users=10)

    state = env.reset()
    processed = preprocess(state)

    state_tensor = torch.tensor(processed, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(state_tensor)
        action = torch.softmax(output, dim=0)

    print("\n📡 Current Network State:")
    print(state)

    print("\n🤖 AI Resource Allocation:")
    print(action.cpu().numpy())


if __name__ == "__main__":
    main()
