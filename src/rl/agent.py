import torch
import torch.nn as nn
import torch.optim as optim
import random

class DDQNAgent:
    """
    DDQN (Double Deep Q-Network) 에이전트 모듈.
    행동을 1차적으로 선택하는 MainNet과, 그 행동의 점수를 평가하는 TargetNet의 파라미터를 역으로 결합시켜
    RL 특유의 '과대적합/과대평가(Overestimation)'를 방지합니다.
    """
    def __init__(self, main_net, target_net, num_actions=15, lr=1e-4, gamma=0.99, device='cuda'):
        # 네트워크 세팅 (위에서 우리가 만든 FullDRLNetwork가 넘어옴)
        self.main_net = main_net.to(device)
        self.target_net = target_net.to(device)
        
        # 하드 카피 (Target 망이 Main 망과 완벽히 똑같은 뇌를 복사하게 만듦)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval() # TargetNet은 평가용이므로 기울기 역전파를 하지 않음
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        
        # 학습 최적화 도구 및 손실 함수(Huber Loss - 폭발적 튕김 방지용) 세팅
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def select_action(self, state, epsilon):
        """
        입실론-그리디(Exploration vs Exploitation) 방식으로 Action 선택.
        자연 로그 곡선처럼 입실론 확률 안에선 무작위 행동, 그 이외엔 MainNet이 제일 유망하다는 행동 선택.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state = state.to(self.device)  # (1, C, H, W) 크기 텐서 입력
                q_values = self.main_net(state) # 각 15개 클래스의 행동가치 예측 (1, 15)
                return torch.argmax(q_values, dim=1).item() # 유망한 값의 Index 리턴

    def update_networks(self, batch, indices, weights, replay_buffer):
        """
        TargetNet과 MainNet 그리고 PER 버퍼에서 추출한 데이터를 총망라해 손실(Loss)을 계산하고 가중치를 수정합니다.
        """
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # 1. MainNet으로 '현재 상태에서 내가 실제로 했던 행동'의 가치 Q(S,A) 계산
        current_q_values = self.main_net(states)
        q_expected = current_q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            # DDQN 핵심 로직
            # 2. MainNet을 이용해 '다음 상태(Next State)에서 가장 좋은 행동(Best Action)'이 뭔지 추론
            next_q_values_main = self.main_net(next_states)
            best_next_actions = torch.argmax(next_q_values_main, dim=1).unsqueeze(1)
            
            # 3. TargetNet을 이용해, 그 'Best Action'의 실제 가치가 얼만지 냉정하게 평가
            next_q_values_target = self.target_net(next_states)
            next_q_target = next_q_values_target.gather(1, best_next_actions).squeeze(1)
            
            # 4. 벨만 방정식(Bellman Equation)을 통한 목표 정답 완성
            q_targets = rewards + (1 - dones) * self.gamma * next_q_target

        # 에러 계산 로직 (TD-Error 산출 -> PER 업데이트 용도)
        td_errors = torch.abs(q_targets - q_expected)
        
        # Loss 구하고, Importance Sampling 가중치 곱해주기
        loss = self.criterion(q_expected, q_targets)
        loss = (loss * weights).mean()

        # 학습(역전파 기울기 전송)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 방금 막 산출된 이 '데이터의 오답률'을 PER 버퍼에 다시 집어넣어 업데이트.
        replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        return loss.item(), td_errors.mean().item()

    def update_target_network(self):
        """
        주기적으로 MainNet (최신 똑똑해진 뇌) 의 완전한 가중치 복사본을 TargetNet에 덮어씌움 (Hard Update)
        """
        self.target_net.load_state_dict(self.main_net.state_dict())
