import os
import yaml
import torch
from tqdm import tqdm
from src.dataset import CXR8Dataset
from src.transforms import get_base_transforms
from src.environment import CXREnv
from src.agent import DDQNAgent

def main():
    # 1. 설정 로드
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = config['env']['device'] if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # 2. 데이터셋 및 환경 초기화
    transforms = get_base_transforms()
    dataset = CXR8Dataset(
        raw_img_dir=config['data']['raw_img_dir'],
        csv_path=config['data']['csv_path'],
        transform=transforms,
        use_bbox_only=True
    )
    
    env = CXREnv(dataset, patch_res=config['data']['img_size'])
    agent = DDQNAgent(action_size=9, in_channels=1, batch_size=config['data']['batch_size'], device=device, config=config)
    
    episodes = config['rl']['total_episodes']
    
    print("Starting DDQN Training Loop...")
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            state = next_state
            total_reward += reward
            
        if e % config['rl']['target_update_freq'] == 0:
            agent.update_target_network()
            
        print(f"Episode: {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Loss: {loss:.4f}")
        
    # 모델 저장
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    save_path = os.path.join(config['paths']['checkpoint_dir'], "best_model.pth")
    torch.save(agent.main_net.state_dict(), save_path)
    print(f"Training Complete. Model saved at {save_path}")

if __name__ == "__main__":
    main()
