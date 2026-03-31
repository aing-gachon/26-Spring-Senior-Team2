import os
import torch
from tqdm import tqdm

from src.utils.utils import set_seed, load_config, setup_logger
from src.data.dataset import get_dataloader
from src.data.transforms import get_transforms
from src.models.networks import FullDRLNetwork
from src.rl.environment import MedicalImageEnv
from src.rl.replay_buffer import PrioritizedReplayBuffer
from src.rl.agent import DDQNAgent

def main():
    # 1. 설정 로드 및 환경 세팅
    config = load_config("configs/config.yaml")
    set_seed(42)
    
    logger = setup_logger("TrainLogger", os.path.join(config['paths']['log_dir'], "train.log"))
    logger.info("Training DDQN Agent Started.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    # 2. 데이터 파이프라인 준비
    transforms = get_transforms(config['env']['image_size'])
    dataloader = get_dataloader(
        img_dir=config['paths']['raw_image_dir'],
        transforms=transforms,
        target_disease=config['env']['target_disease'],
        normal_disease=config['env']['normal_disease'],
        target_count=config['env']['target_disease_count'],
        normal_ratio=config['env']['normal_ratio'],
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True
    )
    
    # 3. 모델, 버퍼, 환경, 에이전트 초기화
    num_classes = config['model']['num_classes']
    main_net = FullDRLNetwork(backbone_name=config['model']['backbone'], num_actions=num_classes)
    target_net = FullDRLNetwork(backbone_name=config['model']['backbone'], num_actions=num_classes)
    
    env = MedicalImageEnv(dataloader=dataloader, num_classes=num_classes, max_steps_per_episode=config['env']['max_steps_per_episode'])
    buffer = PrioritizedReplayBuffer(capacity=config['hyperparameters']['buffer_capacity'])
    
    agent = DDQNAgent(
        main_net=main_net,
        target_net=target_net,
        num_actions=num_classes,
        lr=config['hyperparameters']['learning_rate'],
        gamma=config['hyperparameters']['gamma'],
        device=device
    )
    
    # 4. 하이퍼파라미터 변수화
    num_episodes = config['env']['num_episodes']
    batch_size = config['hyperparameters']['batch_size']
    epsilon = config['hyperparameters']['epsilon_start']
    epsilon_min = config['hyperparameters']['epsilon_end']
    epsilon_decay = config['hyperparameters']['epsilon_decay']
    target_update_interval = config['hyperparameters']['target_update_interval']
    
    global_step = 0
    best_reward = -float('inf')
    
    # 체크포인트 저장용 디렉터리 확인
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    # 5. 본격적인 에피소드 루프 시작
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        # 진행상황 확인용 TQDM 바
        pbar = tqdm(total=env.max_steps, desc=f"Ep [{episode+1}/{num_episodes}]")
        
        while steps < env.max_steps:
            # 상태를 파악하고 액션 취하기
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장 (오답 노트에 밀어넣기)
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # 버퍼에 데이터가 충분히 모이면, 랜덤 추출 후 역전파(학습)
            if len(buffer.buffer) >= batch_size:
                batch, indices, weights = buffer.sample(batch_size)
                loss, _ = agent.update_networks(batch, indices, weights, buffer)
                episode_loss += loss
            
            # 하드카피: 특정 스텝마다 TargetNet 병합
            if global_step % target_update_interval == 0:
                agent.update_target_network()
                
            steps += 1
            global_step += 1
            pbar.update(1)
            
            if done:
                break
                
        pbar.close()
        
        # ε(무작위 탐험 확률) 낮추기
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        avg_loss = episode_loss / steps if steps > 0 else 0
        logger.info(f"Ep {episode+1}: Reward={episode_reward:.1f}, Loss={avg_loss:.4f}, EPS={epsilon:.3f}")
        
        # 최고 성능 모델 수시 저장
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = os.path.join(config['paths']['checkpoint_dir'], "best_model.pth")
            torch.save(agent.main_net.state_dict(), save_path)
            logger.info(f"🚀 New Best Model Saved -> Reward: {best_reward:.1f}")

if __name__ == "__main__":
    main()
