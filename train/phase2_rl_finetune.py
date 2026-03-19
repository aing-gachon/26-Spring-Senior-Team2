import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent import RPPOAgent
from models.classifier import CXRClassifier
from data.env import CXRExplorationEnv
from data.dataset import CXR8Dataset, get_cxr_transforms

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """
    일반화된 어드밴티지 추정(Generalized Advantage Estimation, GAE)을 계산하여 
    분산이 적으면서 정확한 스텝별 반환값(Returns) 배열을 도출합니다.
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train_phase2_ppo():
    """
    Phase 2: RL Fine-Tuning 
    연속적인 PPO 정책 (Continuous Actor-Critic)과 RNN 환경의 탐색 궤적을 토대로 
    탐색 에이전트를 스스로 룰(IoU 증가 위주)에 따라 미세학습(Fine-Tuning) 시킵니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 하이퍼파라미터 (Hyperparameters)
    num_episodes = 500
    max_steps = 15
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = 4
    clip_param = 0.2
    lr = 3e-4
    
    # 1. 환경(gymnasium 인프라) 및 데이터 파이프라인 구성
    root_dir = "C:/Users/qesad/Desktop/tutorials/CXR8"
    bbox_csv = os.path.join(root_dir, "BBox_List_2017.csv")
    entry_csv = os.path.join(root_dir, "Data_Entry_2017_v2020.csv")
    
    dataset = CXR8Dataset(root_dir, bbox_csv, entry_csv, transform=get_cxr_transforms(), use_bbox_only=True)
    env = CXRExplorationEnv(data_list=dataset, max_steps=max_steps)
    
    # 2. 모델 로드 : 에이전트(업데이트 대상) 및 분류기(평가 전용 동결 모델)
    agent = RPPOAgent(in_channels=1).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    classifier = CXRClassifier(in_channels=1, out_classes=14).to(device)
    classifier_path = "C:/Users/qesad/Desktop/tutorials/tumor_aing/checkpoints/phase0_classifier.pth"
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print("Loaded Phase 0 Pre-trained Classifier.")
    else:
        print("Warning: Phase 0 Classifier weight NOT found.")
    
    # 분류 모델 가중치 동결 (Freeze)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    print(f"Starting Phase 2 RL Fine-Tuning (PPO Loop) on device: {device}")
    
    # 3. 강화학습 수집 & 최적화 Loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # 버퍼 객체
        log_probs, values, rewards, masks, states, actions = [], [], [], [], [], []
        
        # 에피소드 초기 RNN Hidden State
        hidden_state = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
        
        episode_reward = 0
        done = False
        
        while not done:
            # 차원 보정 (B=1, Seq=1, C=1, H, W)
            local_patch = torch.FloatTensor(obs["local_patch"]).unsqueeze(0).unsqueeze(0).to(device)
            global_view = torch.FloatTensor(obs["global_view"]).unsqueeze(0).unsqueeze(0).to(device)
            coords = torch.FloatTensor(obs["coords"]).unsqueeze(0).unsqueeze(0).to(device)
            states.append((local_patch, global_view, coords))
            
            # 에이전트 순전파
            mu, logstd, terminal_prob, value, hidden_state = agent(local_patch, global_view, coords, hidden_state)
            
            # 연속 행동 확률 샘플링 (dx, dy, dw, dh)
            std = logstd.exp()
            dist = Normal(mu, std)
            action = dist.sample()
            
            # 확정(종결) 선언용 이산 로직 샘플링
            term_dist = Bernoulli(terminal_prob)
            term_action = term_dist.sample()
            
            # 모델 Tensor Data -> Numpy 포맷(gym Action Space) 변환
            env_action = np.zeros(5, dtype=np.float32)
            env_action[:4] = action.squeeze().cpu().numpy()
            env_action[4] = term_action.squeeze().cpu().item()
            
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Terminal일 시 정밀 분류기에 패치를 전송하여 신뢰도를 구하고 이를 보너스 리워드로 추가합니다.
            if terminated:
                with torch.no_grad():
                    logits = classifier(local_patch.squeeze(1)) # (1, 1, 224, 224) 
                    confidence_score = torch.max(logits, dim=1)[0].item()
                    # 정답을 꽤 확신하는 패치를 가져왔다면 배수(x5)로 점수 지급
                    reward += (confidence_score * 5.0)
            
            action_log_prob = dist.log_prob(action).sum(dim=-1) + term_dist.log_prob(term_action).sum(dim=-1)
            
            log_probs.append(action_log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            masks.append(1.0 - float(done))
            actions.append((action, term_action))
            episode_reward += reward
            
            obs = next_obs
            
        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {episode_reward:.2f} | Steps: {env.current_step}")
        
        # 마지막 Next Value 산출
        local_patch = torch.FloatTensor(obs["local_patch"]).unsqueeze(0).unsqueeze(0).to(device)
        global_view = torch.FloatTensor(obs["global_view"]).unsqueeze(0).unsqueeze(0).to(device)
        coords = torch.FloatTensor(obs["coords"]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, _, next_value, _ = agent(local_patch, global_view, coords, hidden_state)
            next_value = next_value.squeeze()
            
        returns = compute_gae(next_value, rewards, masks, values, gamma, gae_lambda)
        
        returns_tensor = torch.cat(returns).detach()
        values_tensor = torch.stack(values).detach()
        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_log_probs = torch.stack(log_probs).detach()
        
        # --- PPO 업데이트 (Surrogate Clipping Policy) ---
        hidden_state = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
        
        for _ in range(ppo_epochs):
            mu_list, logstd_list, term_prob_list, val_list = [], [], [], []
            h = hidden_state
            
            # RNN State 특징 상 배치 처리를 위해 순차적 연산 및 저장
            for (lp, gv, cd) in states:
                mu, logstd, terminal_prob, value, h = agent(lp, gv, cd, h)
                mu_list.append(mu)
                logstd_list.append(logstd)
                term_prob_list.append(terminal_prob)
                val_list.append(value)
                
            mu = torch.stack(mu_list).squeeze()
            logstd = torch.stack(logstd_list).squeeze(dim=1).squeeze(dim=1)
            terminal_prob = torch.stack(term_prob_list).squeeze()
            new_values = torch.stack(val_list).squeeze()
            
            std = logstd.exp()
            dist = Normal(mu, std)
            term_dist = Bernoulli(terminal_prob)
            
            act_tensors = torch.stack([a[0] for a in actions]).squeeze()
            term_tensors = torch.stack([a[1] for a in actions]).squeeze()
            
            new_log_probs = dist.log_prob(act_tensors).sum(dim=-1) + term_dist.log_prob(term_tensors).squeeze()
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(new_values, returns_tensor)
            entropy_bonus = dist.entropy().mean() + term_dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5) # 그래디언트 폭주 제어
            optimizer.step()
            
        # 일정 에피소드 단위로 Policy 가중치 저장
        if (episode + 1) % 100 == 0:
            os.makedirs("C:/Users/qesad/Desktop/tutorials/tumor_aing/checkpoints", exist_ok=True)
            torch.save(agent.state_dict(), f"C:/Users/qesad/Desktop/tutorials/tumor_aing/checkpoints/phase2_ppo_ep{episode+1}.pth")

if __name__ == "__main__":
    train_phase2_ppo()
