import yaml
import torch
import numpy as np
from src.dataset import CXR8Dataset
from src.transforms import get_base_transforms
from src.environment import CXREnv
from src.agent import DDQNAgent

def evaluate():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = config['env']['device'] if torch.cuda.is_available() else "cpu"
    
    transforms = get_base_transforms()
    # 평가 모드에서는 use_bbox_only 등 테스트셋 분리 로직 필요
    dataset = CXR8Dataset(
        raw_img_dir=config['data']['raw_img_dir'],
        csv_path=config['data']['csv_path'],
        transform=transforms,
        use_bbox_only=True
    )
    
    env = CXREnv(dataset, patch_res=config['data']['img_size'])
    agent = DDQNAgent(action_size=9, in_channels=1, batch_size=config['data']['batch_size'], device=device, config=config)
    
    # 가중치 로드
    model_path = config['paths']['checkpoint_dir'] + "best_model.pth"
    agent.main_net.load_state_dict(torch.load(model_path, map_location=device))
    
    # 평가 시에는 탐험 배제
    agent.epsilon = 0.0
    
    total_rewards = []
    
    for i in range(10): # 초기 10개 에피소드 평가
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
        print(f"Test Episode {i+1} : Total Reward = {total_reward:.2f}")
        total_rewards.append(total_reward)
        
    print(f"Average Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    evaluate()
