import os
import torch
from src.utils import get_config, _set_seed
from src.dataset import load_and_filter_data, get_dataloader
from src.environment import MedicalImageEnv
from src.agent import DDQNAgent
from src.replay_buffer import ReplayBuffer

def train_agent():
    # 1. 환경 및 설정 초기화
    _set_seed(42)
    config = get_config("configs/config.yaml")
    
    # 2. 데이터 준비 및 환경 구성
    data_dicts, class_weights = load_and_filter_data(config['data']['csv_path'], config['data']['image_dir'])
    train_loader = get_dataloader(data_dicts, batch_size=1) # 환경 조회를 위해 배치 1
    env = MedicalImageEnv(train_loader, class_weights)
    
    # 3. 강화학습 DDQN 구성요소 초기화
    num_classes = config['agent']['num_classes']
    agent = DDQNAgent(num_classes=num_classes, config=config)
    replay_buffer = ReplayBuffer(config['agent']['buffer_capacity'])
    
    num_episodes = config['train']['num_episodes']
    batch_size = config['agent']['batch_size']
    target_update_freq = config['agent']['target_update_freq']
    
    global_step = 0
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 학습 루프 시작
    print("🚀 เริ่มฝึกสอน (Starting Training)...") # Placeholder for fun
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        loss_history = []
        
        # 1 에피소드당 X-ray 관찰 수 (단일 분류가 종단점, 여기선 시뮬레이션용 연속화)
        # 본 프로젝트 구조상 한 루프 == 한 X-ray 이미지 예측
        for step in range(config['train']['steps_per_episode']):
            # Phase 4/5: 상태 관찰 및 행동 선택 (a_t)
            action = agent.select_action(state)
            
            # 행동 환경에 투여 -> 보상(r_t) 및 다음 상태 획득
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Replay Buffer에 튜플 저장 (PER 기반으로 확장 여지)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # 버퍼 용량이 차면 학습 시작
            if len(replay_buffer) > batch_size:
                # PER 샘플링
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                # 모델 가중치 업데이트 (MainNet) 및 TD 에러 반환
                loss, td_errors = agent.train_step(states, actions, rewards, next_states, dones, weights)
                
                # TD 에러를 기반으로 버퍼의 우선순위 업데이트
                replay_buffer.update_priorities(indices, td_errors)
                
                loss_history.append(loss)
                agent.decay_epsilon()
                
            # 타깃 네트워크(TargetNet) 동기화
            if global_step % target_update_freq == 0:
                agent.update_target_network()
                
            global_step += 1
            
        # Logging
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {episode_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
        
    torch.save(agent.main_net.state_dict(), os.path.join(save_dir, "best_model.pth"))
    print(f"🎉 Training Complete. Model saved to {save_dir}/best_model.pth")

if __name__ == "__main__":
    train_agent()
