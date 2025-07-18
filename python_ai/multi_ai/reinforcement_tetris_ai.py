#!/usr/bin/env python3
"""
PyTorch DQN 기반 다중 전문화 테트리스 AI 시스템
- HoleFindingAI: 구멍 메우기 전문
- ShapeOptimizingAI: 최적 형태 유지 전문  
- StackOptimizingAI: 균등한 높이 유지 전문
- LineClearingAI: 낮은 층 완성 전문
- StrategicAI: 멀티라인 클리어 전문

🆕 스마트 앙상블 모드: 우선순위 기반 AI 선택 시스템
- 위험도 분석: 게임 오버 위험성 평가
- 공간 활용도: 넓은 공간 활용 최적화
- 형태 적합성: 현재 블록과 보드의 조화
- 생존 우선 전략: 위험 상황에서 생존 액션 우선 선택
- 상황별 우선순위: 안전/위험 상황에 따른 AI 역할 조정
"""
import requests
import time
import numpy as np
import random
from collections import deque
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch 사용 가능")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch가 설치되지 않음. 'pip install torch' 실행 필요")

# PyTorch DQN 네트워크
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=5):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 기본 DQN 에이전트
class BaseDQNAgent:
    def __init__(self, input_size, action_size=5, learning_rate=0.001, 
                 discount_factor=0.95, epsilon=1.0, memory_size=10000):
        self.input_size = input_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # PyTorch 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  사용 디바이스: {self.device}")
        
        # 네트워크 초기화
        self.q_network = DQN(input_size, output_size=action_size).to(self.device)
        self.target_network = DQN(input_size, output_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 메모리
        self.memory = deque(maxlen=memory_size)
        self.update_target_frequency = 100
        self.step_count = 0
        
        # 액션 매핑 (restart는 제외하고 게임 오버 시에만 수동으로 처리)
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        
        # 타겟 네트워크 초기화
        self.update_target_network()
    
    def extract_features(self, state):
        """기본 특징 추출 - 상속받는 클래스에서 오버라이드"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 기본 특징들
        features.extend(self._get_height_features(board))
        features.extend(self._get_hole_features(board))
        features.extend(self._get_line_features(board))
        features.extend(self._get_shape_features(board))
        
        # 게임 메타 정보
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _get_height_features(self, board):
        """높이 관련 특징"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height / 20.0)  # 정규화
        return heights
    
    def _get_hole_features(self, board):
        """구멍 관련 특징"""
        holes_per_col = []
        for col in range(10):
            holes = 0
            block_found = False
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    block_found = True
                elif block_found and (board[row][col] is None or board[row][col] == 0):
                    holes += 1
            holes_per_col.append(holes / 20.0)  # 정규화
        return holes_per_col
    
    def _get_line_features(self, board):
        """라인 관련 특징"""
        features = []
        complete_lines = 0
        almost_complete = 0
        
        for row in range(20):
            filled_cells = sum(1 for cell in board[row] if cell is not None and cell != 0)
            if filled_cells == 10:
                complete_lines += 1
            elif filled_cells >= 8:
                almost_complete += 1
        
        features.append(complete_lines / 4.0)  # 최대 4라인
        features.append(almost_complete / 10.0)
        return features
    
    def _get_shape_features(self, board):
        """형태 관련 특징"""
        features = []
        
        # 표면 거칠기
        heights = [0] * 10
        for col in range(10):
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    heights[col] = 20 - row
                    break
        
        roughness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
        features.append(roughness / 100.0)
        
        # 최대 높이
        max_height = max(heights) if heights else 0
        features.append(max_height / 20.0)
        
        return features
    
    def get_q_values(self, state):
        """현재 상태에서 모든 액션의 Q-values 반환"""
        features = self.extract_features(state)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.squeeze().cpu().numpy()
    
    def get_best_q_value(self, state):
        """현재 상태에서 최고 Q-value 반환"""
        q_values = self.get_q_values(state)
        return np.max(q_values)
    
    def get_action(self, state):
        """액션 선택"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        features = self.extract_features(state)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        state_features = self.extract_features(state)
        next_state_features = self.extract_features(next_state)
        self.memory.append((state_features, action, reward, next_state_features, done))
    
    def calculate_reward(self, prev_state, current_state):
        """기본 보상 계산 (균형잡힌 플레이 유도) - 상속받는 클래스에서 오버라이드"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 점수 증가 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.01
        
        # 라인 클리어 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 10
        
        # 기본적인 균형 유지 보상
        curr_board = current_state.get('board', [])
        if curr_board:
            heights = [0] * 10
            for col in range(10):
                for row in range(20):
                    if curr_board[row][col] is not None and curr_board[row][col] != 0:
                        heights[col] = 20 - row
                        break
            
            # 높이 분산이 적을 때 보상
            if heights:
                height_variance = sum((h - sum(heights)/len(heights))**2 for h in heights) / len(heights)
                if height_variance < 15:
                    reward += (15 - height_variance) * 0.5
                
                # 극단적 높이 차이 패널티
                height_diff = max(heights) - min(heights)
                if height_diff > 12:
                    reward -= (height_diff - 12) * 2
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 50
        
        return reward
    
    def replay(self, batch_size=32):
        """경험 재생 학습"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 타겟 네트워크 업데이트
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath):
        """모델 로드 (차원 불일치 시 새로 초기화)"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # 차원 호환성 검사
                q_network_state = checkpoint['q_network_state_dict']
                current_input_size = self.q_network.network[0].in_features
                saved_input_size = q_network_state['network.0.weight'].shape[1]
                
                if current_input_size != saved_input_size:
                    print(f"⚠️  모델 차원 불일치: 저장된 모델 입력 크기({saved_input_size}) != 현재 모델 입력 크기({current_input_size})")
                    print(f"🔄 새 모델로 초기화합니다: {filepath}")
                    return
                
                self.q_network.load_state_dict(q_network_state)
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.step_count = checkpoint.get('step_count', 0)
                print(f"✅ 모델 로드됨: {filepath}")
                
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print(f"🔄 새 모델로 초기화합니다: {filepath}")
        else:
            print(f"❌ 저장된 모델 없음: {filepath}")

# 1. 구멍 찾기 전문 AI
class HoleFindingAI(BaseDQNAgent):
    def __init__(self):
        super().__init__(input_size=25, learning_rate=0.001)  # 10*2 + 1 + 1 + 3 = 25
        self.name = "HoleFinding"
        
    def extract_features(self, state):
        """구멍 위치와 메우기에 특화된 특징"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 각 열의 구멍 위치 분석
        for col in range(10):
            column_features = self._analyze_column_holes(board, col)
            features.extend(column_features)
        
        # 전체 구멍 통계
        total_holes = sum(self._count_holes_in_column(board, col) for col in range(10))
        features.append(total_holes / 50.0)  # 정규화
        
        # 상단 구멍 개수 (더 중요)
        top_holes = sum(self._count_top_holes(board, col) for col in range(10))
        features.append(top_holes / 20.0)
        
        # 게임 상태
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _analyze_column_holes(self, board, col):
        """특정 열의 구멍 분석"""
        holes = 0
        top_hole_depth = 0
        block_found = False
        
        for row in range(20):
            if board[row][col] is not None and board[row][col] != 0:
                block_found = True
            elif block_found and (board[row][col] is None or board[row][col] == 0):
                holes += 1
                if top_hole_depth == 0:
                    top_hole_depth = row
        
        return [holes / 20.0, top_hole_depth / 20.0]
    
    def _count_holes_in_column(self, board, col):
        """열의 총 구멍 개수"""
        holes = 0
        block_found = False
        for row in range(20):
            if board[row][col] is not None and board[row][col] != 0:
                block_found = True
            elif block_found and (board[row][col] is None or board[row][col] == 0):
                holes += 1
        return holes
    
    def _count_top_holes(self, board, col):
        """상단 절반의 구멍 개수"""
        holes = 0
        block_found = False
        for row in range(10):  # 상단 절반만
            if board[row][col] is not None and board[row][col] != 0:
                block_found = True
            elif block_found and (board[row][col] is None or board[row][col] == 0):
                holes += 1
        return holes
    
    def calculate_reward(self, prev_state, current_state):
        """구멍 메우기에 특화된 보상"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 기본 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.01
        
        # 구멍 감소 보상 (높은 가중치)
        prev_holes = self._count_total_holes(prev_state.get('board', []))
        curr_holes = self._count_total_holes(current_state.get('board', []))
        hole_reduction = prev_holes - curr_holes
        reward += hole_reduction * 20  # 구멍 하나당 20점
        
        # 상단 구멍 감소 추가 보상
        prev_top_holes = self._count_total_top_holes(prev_state.get('board', []))
        curr_top_holes = self._count_total_top_holes(current_state.get('board', []))
        top_hole_reduction = prev_top_holes - curr_top_holes
        reward += top_hole_reduction * 30  # 상단 구멍은 더 중요
        
        # 라인 클리어 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 15
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 100
        
        return reward
    
    def _count_total_holes(self, board):
        """전체 보드의 구멍 개수"""
        if not board:
            return 0
        return sum(self._count_holes_in_column(board, col) for col in range(10))
    
    def _count_total_top_holes(self, board):
        """상단 절반의 총 구멍 개수"""
        if not board:
            return 0
        return sum(self._count_top_holes(board, col) for col in range(10))

# 2. 형태 최적화 전문 AI
class ShapeOptimizingAI(BaseDQNAgent):
    def __init__(self):
        super().__init__(input_size=29, learning_rate=0.001)  # 10 + 4 + 10 + 2 + 3 = 29
        self.name = "ShapeOptimizing"
    
    def extract_features(self, state):
        """최적 형태 유지에 특화된 특징"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 높이 분포
        heights = self._get_column_heights(board)
        features.extend([h / 20.0 for h in heights])  # 10개
        
        # 형태 품질 지표
        features.append(self._calculate_roughness(heights) / 100.0)
        features.append(self._calculate_height_variance(heights) / 100.0)
        features.append(max(heights) / 20.0 if heights else 0)
        features.append(min(heights) / 20.0 if heights else 0)
        
        # 표면 분석
        features.extend(self._analyze_surface_shape(board))  # 10개
        
        # 안정성 분석
        features.append(self._calculate_stability(board))
        features.append(self._calculate_compactness(board))
        
        # 게임 상태
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _get_column_heights(self, board):
        """각 열의 높이"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def _calculate_roughness(self, heights):
        """표면 거칠기"""
        return sum(abs(heights[i] - heights[i+1]) for i in range(9))
    
    def _calculate_height_variance(self, heights):
        """높이 분산"""
        if not heights:
            return 0
        mean_height = sum(heights) / len(heights)
        return sum((h - mean_height) ** 2 for h in heights) / len(heights)
    
    def _analyze_surface_shape(self, board):
        """표면 형태 분석"""
        heights = self._get_column_heights(board)
        features = []
        
        for i in range(10):
            if i == 0:
                slope = heights[1] - heights[0] if len(heights) > 1 else 0
            elif i == 9:
                slope = heights[9] - heights[8]
            else:
                slope = (heights[i+1] - heights[i-1]) / 2
            features.append(slope / 10.0)  # 정규화
        
        return features
    
    def _calculate_stability(self, board):
        """구조 안정성"""
        supported_blocks = 0
        total_blocks = 0
        
        for row in range(1, 20):
            for col in range(10):
                if board[row][col] is not None and board[row][col] != 0:
                    total_blocks += 1
                    # 아래에 블록이 있으면 지지받음
                    if board[row+1][col] is not None and board[row+1][col] != 0:
                        supported_blocks += 1
        
        return supported_blocks / (total_blocks + 1)
    
    def _calculate_compactness(self, board):
        """구조 밀집도"""
        filled_cells = 0
        total_cells = 0
        
        # 블록이 있는 영역만 계산
        min_height = 20
        for col in range(10):
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    min_height = min(min_height, row)
                    break
        
        if min_height == 20:
            return 1.0
        
        for row in range(min_height, 20):
            for col in range(10):
                total_cells += 1
                if board[row][col] is not None and board[row][col] != 0:
                    filled_cells += 1
        
        return filled_cells / (total_cells + 1)
    
    def calculate_reward(self, prev_state, current_state):
        """형태 최적화에 특화된 보상 (균형잡힌 형태 유도)"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 기본 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.01
        
        prev_board = prev_state.get('board', [])
        curr_board = current_state.get('board', [])
        
        prev_heights = self._get_column_heights(prev_board)
        curr_heights = self._get_column_heights(curr_board)
        
        # 형태 개선 보상 (표면 거칠기 개선)
        prev_roughness = self._calculate_roughness(prev_heights)
        curr_roughness = self._calculate_roughness(curr_heights)
        roughness_improvement = prev_roughness - curr_roughness
        reward += roughness_improvement * 3  # 2 → 3으로 증가
        
        # 안정성 개선 보상
        prev_stability = self._calculate_stability(prev_board)
        curr_stability = self._calculate_stability(curr_board)
        stability_improvement = curr_stability - prev_stability
        reward += stability_improvement * 25
        
        # 밀집도 개선 보상
        prev_compactness = self._calculate_compactness(prev_board)
        curr_compactness = self._calculate_compactness(curr_board)
        compactness_improvement = curr_compactness - prev_compactness
        reward += compactness_improvement * 20
        
        # 균형잡힌 형태 보상 (새로 추가)
        height_variance = self._calculate_height_variance(curr_heights)
        if height_variance < 10:  # 적당한 높이 분산일 때 보상
            reward += (10 - height_variance) * 2
        elif height_variance > 20:  # 너무 불균등하면 패널티
            reward -= (height_variance - 20) * 3
        
        # 극단적 편중 방지 (새로 추가)
        max_height = max(curr_heights) if curr_heights else 0
        min_height = min(curr_heights) if curr_heights else 0
        height_diff = max_height - min_height
        if height_diff > 10:  # 10칸 이상 차이나면 패널티
            reward -= (height_diff - 10) * 5
        
        # 좌우 대칭성 장려 (새로 추가)
        left_avg = sum(curr_heights[:5]) / 5 if curr_heights else 0
        right_avg = sum(curr_heights[5:]) / 5 if curr_heights else 0
        symmetry_diff = abs(left_avg - right_avg)
        if symmetry_diff <= 1:  # 좌우 균형이 좋으면 보상
            reward += 10
        elif symmetry_diff > 5:  # 좌우 불균형이 심하면 패널티
            reward -= symmetry_diff * 2
        
        # 라인 클리어 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 15
        
        # 높이 패널티
        if max_height > 15:
            reward -= (max_height - 15) * 3
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 100
        
        return reward

# 3. 스택 최적화 전문 AI  
class StackOptimizingAI(BaseDQNAgent):
    def __init__(self):
        super().__init__(input_size=19, learning_rate=0.001)  # 10 + 3 + 2 + 1 + 3 = 19
        self.name = "StackOptimizing"
    
    def extract_features(self, state):
        """균등한 스택 높이 유지에 특화된 특징"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 높이 분포
        heights = self._get_column_heights(board)
        features.extend([h / 20.0 for h in heights])  # 10개
        
        # 높이 통계
        mean_height = sum(heights) / len(heights) if heights else 0
        features.append(mean_height / 20.0)
        features.append(self._calculate_height_std(heights) / 10.0)
        features.append((max(heights) - min(heights)) / 20.0 if heights else 0)
        
        # 높이 균등성
        features.append(self._calculate_evenness_score(heights))
        features.append(self._calculate_balance_score(heights))
        
        # 위험 지표
        features.append(self._calculate_danger_score(heights))
        
        # 게임 상태
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _get_column_heights(self, board):
        """각 열의 높이"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def _calculate_height_std(self, heights):
        """높이 표준편차"""
        if not heights:
            return 0
        mean = sum(heights) / len(heights)
        variance = sum((h - mean) ** 2 for h in heights) / len(heights)
        return variance ** 0.5
    
    def _calculate_evenness_score(self, heights):
        """높이 균등성 점수 (0-1, 1이 가장 균등)"""
        if not heights:
            return 1.0
        
        max_diff = max(heights) - min(heights)
        return max(0, 1 - max_diff / 20.0)
    
    def _calculate_balance_score(self, heights):
        """좌우 균형 점수"""
        if len(heights) < 10:
            return 0.5
        
        left_avg = sum(heights[:5]) / 5
        right_avg = sum(heights[5:]) / 5
        diff = abs(left_avg - right_avg)
        return max(0, 1 - diff / 10.0)
    
    def _calculate_danger_score(self, heights):
        """위험도 점수 (높을수록 위험)"""
        dangerous_heights = sum(1 for h in heights if h > 15)
        return dangerous_heights / 10.0
    
    def calculate_reward(self, prev_state, current_state):
        """스택 최적화에 특화된 보상 (한쪽 쌓기 방지 강화)"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 기본 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.01
        
        prev_board = prev_state.get('board', [])
        curr_board = current_state.get('board', [])
        
        prev_heights = self._get_column_heights(prev_board)
        curr_heights = self._get_column_heights(curr_board)
        
        # 균등성 개선 보상 (강화)
        prev_evenness = self._calculate_evenness_score(prev_heights)
        curr_evenness = self._calculate_evenness_score(curr_heights)
        evenness_improvement = curr_evenness - prev_evenness
        reward += evenness_improvement * 100  # 50 → 100으로 증가
        
        # 균형 개선 보상 (강화)
        prev_balance = self._calculate_balance_score(prev_heights)
        curr_balance = self._calculate_balance_score(curr_heights)
        balance_improvement = curr_balance - prev_balance
        reward += balance_improvement * 60  # 30 → 60으로 증가
        
        # 위험도 감소 보상
        prev_danger = self._calculate_danger_score(prev_heights)
        curr_danger = self._calculate_danger_score(curr_heights)
        danger_reduction = prev_danger - curr_danger
        reward += danger_reduction * 40
        
        # 한쪽 편중 방지 패널티 (새로 추가)
        height_variance = self._calculate_height_std(curr_heights)
        if height_variance > 5:  # 높이 차이가 클 때 패널티
            reward -= height_variance * 5
        
        # 극단적 높이 차이 패널티 (새로 추가)
        max_height = max(curr_heights) if curr_heights else 0
        min_height = min(curr_heights) if curr_heights else 0
        extreme_diff = max_height - min_height
        if extreme_diff > 8:  # 8칸 이상 차이나면 강한 패널티
            reward -= (extreme_diff - 8) * 15
        
        # 가운데 영역 사용 장려 (새로 추가)
        center_heights = curr_heights[3:7]  # 가운데 4열
        edge_heights = curr_heights[:3] + curr_heights[7:]  # 양쪽 끝 6열
        center_avg = sum(center_heights) / len(center_heights) if center_heights else 0
        edge_avg = sum(edge_heights) / len(edge_heights) if edge_heights else 0
        
        # 가운데가 너무 높으면 패널티, 적절히 사용하면 보상
        if center_avg > edge_avg + 2:
            reward -= 10
        elif abs(center_avg - edge_avg) <= 2:
            reward += 5
        
        # 라인 클리어 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 20
        
        # 최대 높이 패널티
        if max_height > 18:
            reward -= (max_height - 18) * 10
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 100
        
        return reward

# 4. 라인 클리어 전문 AI
class LineClearingAI(BaseDQNAgent):
    def __init__(self):
        super().__init__(input_size=25, learning_rate=0.001)  # 5 + 5 + 1 + 1 + 10 + 3 = 25
        self.name = "LineClearing"
    
    def extract_features(self, state):
        """낮은 층 완성에 특화된 특징"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 각 행의 채워진 정도 (하단부터)
        for row in range(15, 20):  # 하단 5줄
            filled_count = sum(1 for cell in board[row] if cell is not None and cell != 0)
            features.append(filled_count / 10.0)
        
        # 각 행의 빈 칸 패턴
        for row in range(15, 20):
            empty_positions = []
            for col in range(10):
                if board[row][col] is None or board[row][col] == 0:
                    empty_positions.append(col)
            
            # 빈 칸의 연속성
            continuity = self._calculate_continuity(empty_positions)
            features.append(continuity)
        
        # 완성 가능한 라인 수
        almost_complete = 0
        for row in range(20):
            filled = sum(1 for cell in board[row] if cell is not None and cell != 0)
            if filled >= 8:
                almost_complete += 1
        features.append(almost_complete / 10.0)
        
        # 하단 밀집도
        bottom_density = self._calculate_bottom_density(board)
        features.append(bottom_density)
        
        # 각 열의 높이
        heights = self._get_column_heights(board)
        features.extend([h / 20.0 for h in heights])
        
        # 게임 상태
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _calculate_continuity(self, empty_positions):
        """빈 칸의 연속성 계산"""
        if not empty_positions:
            return 1.0
        if len(empty_positions) == 1:
            return 0.5
        
        consecutive_groups = 0
        current_group_size = 1
        
        for i in range(1, len(empty_positions)):
            if empty_positions[i] == empty_positions[i-1] + 1:
                current_group_size += 1
            else:
                consecutive_groups += 1
                current_group_size = 1
        consecutive_groups += 1
        
        # 그룹이 적을수록 좋음
        return max(0, 1 - consecutive_groups / 5.0)
    
    def _calculate_bottom_density(self, board):
        """하단 5줄의 밀집도"""
        total_cells = 50  # 5행 * 10열
        filled_cells = 0
        
        for row in range(15, 20):
            for col in range(10):
                if board[row][col] is not None and board[row][col] != 0:
                    filled_cells += 1
        
        return filled_cells / total_cells
    
    def _get_column_heights(self, board):
        """각 열의 높이"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def calculate_reward(self, prev_state, current_state):
        """라인 클리어에 특화된 보상"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 기본 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.02
        
        # 라인 클리어 보상 (매우 높은 가중치)
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 100
        
        prev_board = prev_state.get('board', [])
        curr_board = current_state.get('board', [])
        
        # 하단 밀집도 개선 보상
        prev_density = self._calculate_bottom_density(prev_board)
        curr_density = self._calculate_bottom_density(curr_board)
        density_improvement = curr_density - prev_density
        reward += density_improvement * 50
        
        # 거의 완성된 라인 생성 보상
        prev_almost = self._count_almost_complete_lines(prev_board)
        curr_almost = self._count_almost_complete_lines(curr_board)
        almost_improvement = curr_almost - prev_almost
        reward += almost_improvement * 20
        
        # 하단 행 완성도 개선 보상
        for row in range(17, 20):  # 하단 3줄
            prev_filled = sum(1 for cell in prev_board[row] if cell is not None and cell != 0)
            curr_filled = sum(1 for cell in curr_board[row] if cell is not None and cell != 0)
            row_improvement = curr_filled - prev_filled
            reward += row_improvement * 5 * (21 - row)  # 더 아래일수록 높은 보상
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 100
        
        return reward
    
    def _count_almost_complete_lines(self, board):
        """거의 완성된 라인 수"""
        count = 0
        for row in range(20):
            filled = sum(1 for cell in board[row] if cell is not None and cell != 0)
            if filled >= 8:
                count += 1
        return count

# 5. 전략적 멀티라인 클리어 AI
class StrategicAI(BaseDQNAgent):
    def __init__(self):
        super().__init__(input_size=36, learning_rate=0.001)  # 20 + 3 + 1 + 9 + 3 = 36
        self.name = "Strategic"
    
    def extract_features(self, state):
        """전략적 멀티라인 클리어에 특화된 특징"""
        if not state or 'board' not in state:
            return np.zeros(self.input_size)
        
        board = state['board']
        features = []
        
        # 각 행의 완성도
        for row in range(20):
            filled = sum(1 for cell in board[row] if cell is not None and cell != 0)
            features.append(filled / 10.0)
        
        # 멀티라인 가능성 분석
        features.append(self._analyze_tetris_potential(board))
        features.append(self._analyze_triple_potential(board))
        features.append(self._analyze_double_potential(board))
        
        # 웰 (깊은 구멍) 분석
        well_depth = self._find_deepest_well(board)
        features.append(well_depth / 20.0)
        
        # 스택 패턴 분석
        features.extend(self._analyze_stack_pattern(board))
        
        # 게임 상태
        features.append(state.get('score', 0) / 10000.0)
        features.append(state.get('lines', 0) / 100.0)
        features.append(state.get('level', 1) / 20.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _analyze_tetris_potential(self, board):
        """테트리스 (4라인) 가능성"""
        # 4줄이 연속으로 거의 완성되어 있고, 한 열이 비어있는 패턴
        for start_row in range(17):  # 4줄 확인 가능한 시작점
            potential_cols = []
            for col in range(10):
                col_empty_in_4_rows = True
                for row in range(start_row, start_row + 4):
                    if board[row][col] is not None and board[row][col] != 0:
                        col_empty_in_4_rows = False
                        break
                
                if col_empty_in_4_rows:
                    # 이 열이 4줄에서 모두 비어있는지 확인
                    other_cols_full = True
                    for check_col in range(10):
                        if check_col != col:
                            for row in range(start_row, start_row + 4):
                                if board[row][check_col] is None or board[row][check_col] == 0:
                                    other_cols_full = False
                                    break
                    if other_cols_full:
                        return 1.0
        return 0.0
    
    def _analyze_triple_potential(self, board):
        """3라인 클리어 가능성"""
        potential_count = 0
        for start_row in range(18):  # 3줄 확인
            almost_complete = 0
            for row in range(start_row, start_row + 3):
                filled = sum(1 for cell in board[row] if cell is not None and cell != 0)
                if filled >= 9:
                    almost_complete += 1
            if almost_complete >= 2:
                potential_count += 1
        return min(potential_count / 5.0, 1.0)
    
    def _analyze_double_potential(self, board):
        """2라인 클리어 가능성"""
        potential_count = 0
        for start_row in range(19):  # 2줄 확인
            almost_complete = 0
            for row in range(start_row, start_row + 2):
                filled = sum(1 for cell in board[row] if cell is not None and cell != 0)
                if filled >= 8:
                    almost_complete += 1
            if almost_complete == 2:
                potential_count += 1
        return min(potential_count / 10.0, 1.0)
    
    def _find_deepest_well(self, board):
        """가장 깊은 웰 찾기"""
        heights = self._get_column_heights(board)
        max_well_depth = 0
        
        for col in range(10):
            # 양쪽 이웃보다 낮은 열이 웰
            left_height = heights[col-1] if col > 0 else 0
            right_height = heights[col+1] if col < 9 else 0
            current_height = heights[col]
            
            well_depth = min(left_height, right_height) - current_height
            if well_depth > 0:
                max_well_depth = max(max_well_depth, well_depth)
        
        return max_well_depth
    
    def _analyze_stack_pattern(self, board):
        """스택 패턴 분석"""
        heights = self._get_column_heights(board)
        patterns = []
        
        # 경사 패턴 (테트리스 준비용)
        for i in range(7):  # 4열 연속 확인
            slope_pattern = all(heights[i+j] >= heights[i+j+1] for j in range(3))
            patterns.append(1.0 if slope_pattern else 0.0)
        
        # 안정성 패턴
        stable_pattern = all(abs(heights[i] - heights[i+1]) <= 2 for i in range(9))
        patterns.append(1.0 if stable_pattern else 0.0)
        
        # 웰 패턴 (한쪽이 깊게 파인 패턴)
        well_pattern = any(heights[i] < heights[i-1] - 3 and heights[i] < heights[i+1] - 3 
                          for i in range(1, 9))
        patterns.append(1.0 if well_pattern else 0.0)
        
        return patterns
    
    def _get_column_heights(self, board):
        """각 열의 높이"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def calculate_reward(self, prev_state, current_state):
        """전략적 플레이에 특화된 보상"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 기본 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.02
        
        # 멀티라인 클리어 보상 (기하급수적 증가)
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        if lines_diff == 4:  # 테트리스
            reward += 1000
        elif lines_diff == 3:  # 트리플
            reward += 300
        elif lines_diff == 2:  # 더블
            reward += 100
        elif lines_diff == 1:  # 싱글
            reward += 25
        
        prev_board = prev_state.get('board', [])
        curr_board = current_state.get('board', [])
        
        # 테트리스 준비 보상
        prev_tetris_potential = self._analyze_tetris_potential(prev_board)
        curr_tetris_potential = self._analyze_tetris_potential(curr_board)
        tetris_setup_improvement = curr_tetris_potential - prev_tetris_potential
        reward += tetris_setup_improvement * 200
        
        # 전략적 패턴 보상
        prev_patterns = self._analyze_stack_pattern(prev_board)
        curr_patterns = self._analyze_stack_pattern(curr_board)
        pattern_improvement = sum(curr_patterns) - sum(prev_patterns)
        reward += pattern_improvement * 50
        
        # 웰 관리 보상
        prev_well = self._find_deepest_well(prev_board)
        curr_well = self._find_deepest_well(curr_board)
        if curr_well > prev_well and curr_well >= 4:  # 깊은 웰 생성
            reward += 100
        elif curr_well < prev_well and prev_well >= 4:  # 웰 사용
            reward += 150
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 150
        
        return reward

# API 클라이언트
class TetrisAPIClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/game"
    
    def get_game_state(self):
        try:
            response = requests.get(self.api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {})
        except:
            pass
        return None
    
    def send_action(self, action):
        try:
            response = requests.post(self.api_url, json={
                'type': 'action', 
                'action': action
            }, timeout=5)
            return response.status_code == 200
        except:
            return False

# 멀티 AI 시스템
class MultiAITetrisSystem:
    def __init__(self):
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch가 필요합니다. 'pip install torch' 실행 후 다시 시도하세요.")
            return
        
        self.client = TetrisAPIClient()
        
        # 5개의 전문화된 AI 초기화
        self.ais = {
            'hole_finding': HoleFindingAI(),
            'shape_optimizing': ShapeOptimizingAI(), 
            'stack_optimizing': StackOptimizingAI(),
            'line_clearing': LineClearingAI(),
            'strategic': StrategicAI()
        }
        
        # 모델 로드
        for name, ai in self.ais.items():
            model_path = f"models/{name}_model.pth"
            os.makedirs("models", exist_ok=True)
            ai.load_model(model_path)
        
        print(f"✅ {len(self.ais)}개의 전문화된 AI 로드 완료")
    
    def train_all(self, episodes=50, max_steps=500):
        """모든 AI 훈련"""
        print("🧠 다중 AI 훈련 시작!")
        print(f"📊 각 AI당 {episodes} 에피소드, 최대 {max_steps} 스텝")
        print("=" * 60)
        
        for ai_name, ai in self.ais.items():
            print(f"\n🤖 {ai_name.upper()} AI 훈련 시작")
            print("-" * 40)
            self._train_single_ai(ai, ai_name, episodes, max_steps)
            
            # 모델 저장
            model_path = f"models/{ai_name}_model.pth"
            ai.save_model(model_path)
            print(f"💾 {ai_name} 모델 저장 완료")
            
            # AI 간 휴식
            print("⏳ 다음 AI 훈련까지 5초 대기...")
            time.sleep(5)
        
        print("\n✅ 모든 AI 훈련 완료!")
    
    def _train_single_ai(self, ai, ai_name, episodes, max_steps):
        """단일 AI 훈련"""
        for episode in range(episodes):
            print(f"📈 에피소드 {episode + 1}/{episodes}")
            
            total_reward = 0
            step = 0
            
            # 게임 초기화
            prev_state = self.client.get_game_state()
            if not prev_state:
                print("❌ 서버 연결 실패")
                continue
            
            for step in range(max_steps):
                current_state = self.client.get_game_state()
                if not current_state:
                    break
                
                # 게임 오버 처리 (강화된 감지)
                is_game_over = current_state.get('isGameOver', False)
                
                # 추가적인 게임 오버 감지 조건들
                if not is_game_over:
                    board = current_state.get('board', [])
                    if board:
                        # 상단 3줄에 블록이 있는지 확인
                        for row in range(3):
                            if any(board[row][col] is not None and board[row][col] != 0 for col in range(10)):
                                is_game_over = True
                                print(f"   🚨 AI가 게임 오버 감지: 상단 영역 블록 발견 (행 {row})")
                                break
                
                if is_game_over:
                    print(f"   💀 게임 오버 감지! 점수: {current_state.get('score', 0)}, 라인: {current_state.get('lines', 0)}")
                    
                    print(f"   🔄 즉시 재시작 시도...")
                    if self.client.send_action('restart'):
                        print(f"   ✅ 재시작 요청 전송됨")
                        time.sleep(1.5)  # 재시작 처리 대기
                        
                        # 재시작 성공 확인 (여러 번 시도)
                        for attempt in range(3):
                            time.sleep(0.5)
                            new_state = self.client.get_game_state()
                            if new_state and not new_state.get('isGameOver', False) and new_state.get('score', 0) == 0:
                                print(f"   🎮 새 게임 시작 확인 (시도 {attempt + 1})")
                                break
                        else:
                            print(f"   ⚠️ 재시작 확인 실패, 다음 에피소드로...")
                            break
                        continue
                    else:
                        print(f"   ❌ 재시작 요청 실패")
                        break
                
                # AI 액션 선택 및 실행
                action_idx = ai.get_action(current_state)
                action = ai.actions[action_idx]
                
                if not self.client.send_action(action):
                    continue
                
                time.sleep(0.3)
                
                # 다음 상태 및 보상
                next_state = self.client.get_game_state()
                if not next_state:
                    continue
                
                reward = ai.calculate_reward(current_state, next_state)
                total_reward += reward
                
                # 경험 저장
                done = next_state.get('isGameOver', False)
                ai.remember(current_state, action_idx, reward, next_state, done)
                
                # 학습
                ai.replay()
                
                # 진행 상황
                if step % 50 == 0:
                    score = next_state.get('score', 0)
                    lines = next_state.get('lines', 0)
                    print(f"   📊 스텝 {step}: 점수 {score}, 라인 {lines}, 보상 {reward:.2f}")
                
                prev_state = current_state
            
            # 에피소드 결과
            final_state = self.client.get_game_state()
            final_score = final_state.get('score', 0) if final_state else 0
            final_lines = final_state.get('lines', 0) if final_state else 0
            
            print(f"   🎯 완료: 점수 {final_score}, 라인 {final_lines}, 총보상 {total_reward:.2f}")
            print(f"   🔍 탐험률: {ai.epsilon:.3f}")
    
    def play_with_ai(self, ai_name, duration=120):
        """특정 AI로 플레이"""
        if ai_name not in self.ais:
            print(f"❌ AI '{ai_name}' 없음. 사용 가능: {list(self.ais.keys())}")
            return
        
        ai = self.ais[ai_name]
        print(f"🤖 {ai_name.upper()} AI 플레이 시작!")
        print(f"🕐 플레이 시간: {duration}초")
        print("=" * 40)
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            state = self.client.get_game_state()
            if not state:
                time.sleep(0.5)
                continue
            
            # 게임 오버 처리 (강화된 감지)
            is_game_over = state.get('isGameOver', False)
            
            # 추가적인 게임 오버 감지 조건들
            if not is_game_over:
                board = state.get('board', [])
                if board:
                    # 상단 3줄에 블록이 있는지 확인
                    for row in range(3):
                        if any(board[row][col] is not None and board[row][col] != 0 for col in range(10)):
                            is_game_over = True
                            print(f"🚨 AI가 게임 오버 감지: 상단 영역 블록 발견 (행 {row})")
                            break
            
            if is_game_over:
                print(f"🎯 게임 오버 감지! 점수: {state.get('score', 0)}, 라인: {state.get('lines', 0)}")
                
                print(f"🔄 즉시 재시작 시도...")
                if self.client.send_action('restart'):
                    print(f"✅ 재시작 요청 전송됨")
                    time.sleep(1.5)  # 재시작 처리 대기
                    
                    # 재시작 성공 확인 (여러 번 시도)
                    for attempt in range(3):
                        time.sleep(0.5)
                        new_state = self.client.get_game_state()
                        if new_state and not new_state.get('isGameOver', False) and new_state.get('score', 0) == 0:
                            print(f"🎮 새 게임 시작 확인 (시도 {attempt + 1}), 계속 플레이...")
                            break
                    else:
                        print(f"⚠️ 재시작 확인 실패")
                        break
                    continue
                else:
                    print(f"❌ 재시작 요청 실패")
                    break
            
            # 최적 액션 선택 (탐험 없이)
            old_epsilon = ai.epsilon
            ai.epsilon = 0
            action_idx = ai.get_action(state)
            action = ai.actions[action_idx]
            ai.epsilon = old_epsilon
            
            if self.client.send_action(action):
                step += 1
                if step % 20 == 0:
                    score = state.get('score', 0)
                    lines = state.get('lines', 0)
                    print(f"🎮 스텝 {step}: 점수 {score}, 라인 {lines}, 액션 {action}")
            
            time.sleep(0.4)
        
        print(f"✅ 플레이 완료! 총 {step} 액션")
    
    def analyze_game_situation(self, state):
        """게임 상황 분석 (위험도, 공간 활용도, 형태 적합성)"""
        if not state or 'board' not in state:
            return {'danger_level': 0, 'space_utilization': 0, 'shape_fitness': 0}
        
        board = state['board']
        analysis = {}
        
        # 1. 위험도 평가 (0-10, 높을수록 위험)
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        
        max_height = max(heights) if heights else 0
        height_variance = sum((h - sum(heights)/len(heights))**2 for h in heights) / len(heights) if heights else 0
        
        danger_level = 0
        if max_height > 15:  # 상단 접근
            danger_level += (max_height - 15) * 2
        if height_variance > 25:  # 불균등한 높이
            danger_level += height_variance * 0.2
        if max_height > 18:  # 매우 위험
            danger_level += 5
        
        analysis['danger_level'] = min(danger_level, 10)
        
        # 2. 공간 활용도 평가 (0-10, 높을수록 좋음)
        # 넓은 공간이 있는지 확인
        wide_spaces = 0
        for col in range(10):
            if heights[col] < max_height - 3:  # 상대적으로 낮은 공간
                wide_spaces += 1
        
        # 구멍 개수 (적을수록 좋음)
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    block_found = True
                elif block_found and (board[row][col] is None or board[row][col] == 0):
                    holes += 1
        
        space_score = wide_spaces * 2 - holes * 0.5
        analysis['space_utilization'] = max(0, min(space_score, 10))
        
        # 3. 형태 적합성 평가 (0-10, 높을수록 좋음)
        # 표면 평탄도
        roughness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
        smoothness = max(0, 10 - roughness * 0.5)
        
        # 좌우 균형
        left_avg = sum(heights[:5]) / 5
        right_avg = sum(heights[5:]) / 5
        balance = max(0, 10 - abs(left_avg - right_avg) * 2)
        
        analysis['shape_fitness'] = (smoothness + balance) / 2
        
        return analysis
    
    def analyze_current_piece_fit(self, state):
        """현재 블록이 보드에 얼마나 잘 맞는지 분석"""
        current_piece = state.get('currentPiece', {})
        if not current_piece:
            return {'fit_score': 5.0, 'best_position': None}
        
        board = state.get('board', [])
        if not board:
            return {'fit_score': 5.0, 'best_position': None}
        
        # 현재 블록의 형태와 위치 분석
        piece_type = current_piece.get('type', '')
        piece_rotation = current_piece.get('rotation', 0)
        
        # 각 위치에서의 적합도 계산
        best_fit = 0
        best_position = None
        
        for col in range(8):  # 가능한 열 위치
            fit_score = self._calculate_piece_fit(board, piece_type, col, piece_rotation)
            if fit_score > best_fit:
                best_fit = fit_score
                best_position = col
        
        return {'fit_score': best_fit, 'best_position': best_position}
    
    def _calculate_piece_fit(self, board, piece_type, col, rotation):
        """특정 위치에서 블록의 적합도 계산"""
        # 간단한 휴리스틱으로 적합도 계산
        # 실제 구현에서는 더 정교한 계산이 필요
        
        # 열 높이 확인
        heights = []
        for c in range(10):
            height = 0
            for row in range(20):
                if board[row][c] is not None and board[row][c] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        
        # 해당 위치의 높이
        target_height = heights[col] if col < len(heights) else 0
        
        # 적합도 점수 계산
        fit_score = 5.0  # 기본 점수
        
        # 너무 높은 위치는 감점
        if target_height > 15:
            fit_score -= (target_height - 15) * 0.5
        
        # 주변 높이와의 조화
        if col > 0 and col < 9:
            left_height = heights[col - 1]
            right_height = heights[col + 1]
            height_diff = abs(left_height - target_height) + abs(right_height - target_height)
            fit_score -= height_diff * 0.1
        
        return max(0, fit_score)
    
    def get_ai_priority(self, ai_name, situation, piece_analysis=None):
        """상황과 현재 블록을 고려한 AI 우선순위 결정"""
        danger = situation['danger_level']
        space = situation['space_utilization']
        shape = situation['shape_fitness']
        
        priorities = {
            'hole_finding': 1.0,      # 기본 우선순위
            'shape_optimizing': 1.0,
            'stack_optimizing': 1.0,
            'line_clearing': 1.0,
            'strategic': 1.0
        }
        
        # 현재 블록 적합도 고려
        if piece_analysis and piece_analysis['fit_score'] < 3:
            # 블록이 잘 맞지 않는 경우 형태 최적화 우선
            priorities['shape_optimizing'] *= 1.5
            priorities['stack_optimizing'] *= 1.3
        
        # 위험 상황 (danger > 6): 생존 전략 우선
        if danger > 6:
            priorities['stack_optimizing'] = 3.0  # 균등한 높이 유지 최우선
            priorities['hole_finding'] = 2.5      # 구멍 메우기 우선
            priorities['line_clearing'] = 2.0     # 라인 클리어 우선
            priorities['shape_optimizing'] = 1.5  # 형태 정리
            priorities['strategic'] = 0.5         # 전략적 플레이 후순위
            
            # 극도로 위험한 상황에서는 더 보수적으로
            if danger > 8:
                priorities['stack_optimizing'] = 4.0
                priorities['strategic'] = 0.2
        
        # 중간 위험 (3 < danger <= 6): 균형 잡힌 전략
        elif danger > 3:
            priorities['stack_optimizing'] = 2.0
            priorities['shape_optimizing'] = 1.8
            priorities['hole_finding'] = 1.5
            priorities['line_clearing'] = 1.3
            priorities['strategic'] = 1.0
        
        # 안전 상황 (danger <= 3): 공간 활용도에 따른 전략
        else:
            if space > 7:  # 넓은 공간 많음
                priorities['strategic'] = 2.5     # 전략적 플레이 우선
                priorities['line_clearing'] = 2.0 # 라인 클리어 준비
                priorities['shape_optimizing'] = 1.5
                priorities['stack_optimizing'] = 1.2
                priorities['hole_finding'] = 1.0
            elif space < 4:  # 공간 부족
                priorities['hole_finding'] = 2.0  # 구멍 메우기 우선
                priorities['shape_optimizing'] = 1.8
                priorities['stack_optimizing'] = 1.5
                priorities['line_clearing'] = 1.2
                priorities['strategic'] = 1.0
            else:  # 보통 공간
                if shape < 5:  # 형태 불량
                    priorities['shape_optimizing'] = 2.0
                    priorities['stack_optimizing'] = 1.5
                else:  # 형태 양호
                    priorities['strategic'] = 1.8
                    priorities['line_clearing'] = 1.5
        
        return priorities[ai_name]
    
    def ensemble_action(self, state):
        """우선순위 기반 AI 선택 시스템"""
        ai_evaluations = {}
        
        # 게임 상황 분석
        situation = self.analyze_game_situation(state)
        
        # 현재 블록 적합도 분석
        piece_analysis = self.analyze_current_piece_fit(state)
        
        for ai_name, ai in self.ais.items():
            try:
                # 탐험 없이 평가
                old_epsilon = ai.epsilon
                ai.epsilon = 0
                
                # Q-value 계산
                best_q_value = ai.get_best_q_value(state)
                action_idx = ai.get_action(state)
                action = ai.actions[action_idx]
                
                # 우선순위 적용 (현재 블록 분석 포함)
                priority = self.get_ai_priority(ai_name, situation, piece_analysis)
                
                # 위험 상황에서는 생존 전략 강화
                if situation['danger_level'] > 6:
                    # 위험한 액션 패널티 (drop, rotate는 신중하게)
                    if action in ['drop'] and situation['space_utilization'] < 5:
                        priority *= 0.5  # 공간 부족 시 drop 패널티
                    elif action in ['rotate'] and situation['shape_fitness'] < 3:
                        priority *= 0.7  # 형태 불량 시 rotate 패널티
                    
                    # 넓은 공간 활용 우선 (사용자 요청 반영)
                    if situation['space_utilization'] > 6:
                        if action in ['left', 'right']:
                            priority *= 1.5  # 넓은 공간으로 이동 장려
                        elif action in ['drop'] and ai_name == 'stack_optimizing':
                            priority *= 0.8  # 넓은 공간이 있을 때 급하게 떨어뜨리지 않기
                
                # 중간 위험 상황에서도 공간 활용 고려
                elif situation['danger_level'] > 3:
                    if situation['space_utilization'] > 7:
                        if action in ['left', 'right']:
                            priority *= 1.3  # 공간 활용 위치 조정 장려
                        elif action in ['rotate'] and piece_analysis['fit_score'] > 5:
                            priority *= 1.2  # 좋은 공간에서 맞춤 회전 장려
                
                # 현재 블록 적합도에 따른 추가 조정
                if piece_analysis['fit_score'] < 2:
                    # 블록이 매우 잘 맞지 않는 경우
                    if action in ['left', 'right']:
                        priority *= 1.2  # 위치 조정 액션 우선
                    elif action in ['drop']:
                        priority *= 0.3  # 급하게 떨어뜨리기 지양
                elif piece_analysis['fit_score'] > 7:
                    # 블록이 잘 맞는 경우
                    if action in ['drop']:
                        priority *= 1.3  # 빠르게 놓기 장려
                
                # 최종 점수 = Q-value * 우선순위
                final_score = best_q_value * priority
                
                ai_evaluations[ai_name] = {
                    'q_value': best_q_value,
                    'priority': priority,
                    'final_score': final_score,
                    'action': action,
                    'action_idx': action_idx
                }
                
                # 원래 epsilon 복원
                ai.epsilon = old_epsilon
                
            except Exception as e:
                print(f"⚠️ {ai_name} 평가 실패: {e}")
                continue
        
        if not ai_evaluations:
            return None, None, None, None
        
        # 최고 최종 점수를 가진 AI 선택
        best_ai_name = max(ai_evaluations.keys(), key=lambda name: ai_evaluations[name]['final_score'])
        best_evaluation = ai_evaluations[best_ai_name]
        
        return best_ai_name, best_evaluation['action'], ai_evaluations, situation
    
    def play_with_ensemble(self, duration=120):
        """스마트 앙상블 방식으로 플레이 (우선순위 기반 AI 선택)"""
        print("🧠 스마트 앙상블 AI 플레이 시작!")
        print("📊 상황별 우선순위 기반 AI 선택 시스템")
        print("🎯 생존 우선 전략: 위험 상황에서 생존 액션 우선 선택")
        print("🔍 공간 활용 최적화: 넓은 공간 활용 우선, 형태 맞춤 회전 장려")
        print(f"🕐 플레이 시간: {duration}초")
        print("=" * 60)
        
        start_time = time.time()
        step = 0
        ai_selection_count = {name: 0 for name in self.ais.keys()}
        
        while time.time() - start_time < duration:
            state = self.client.get_game_state()
            if not state:
                time.sleep(0.5)
                continue
            
            # 게임 오버 처리 (강화된 감지)
            is_game_over = state.get('isGameOver', False)
            
            # 추가적인 게임 오버 감지 조건들
            if not is_game_over:
                board = state.get('board', [])
                if board:
                    # 상단 3줄에 블록이 있는지 확인
                    for row in range(3):
                        if any(board[row][col] is not None and board[row][col] != 0 for col in range(10)):
                            is_game_over = True
                            print(f"🚨 앙상블 AI가 게임 오버 감지: 상단 영역 블록 발견 (행 {row})")
                            break
            
            if is_game_over:
                print(f"💀 게임 오버 감지! 점수: {state.get('score', 0)}, 라인: {state.get('lines', 0)}")
                
                print(f"🔄 즉시 재시작 시도...")
                if self.client.send_action('restart'):
                    print(f"✅ 재시작 요청 전송됨")
                    time.sleep(1.5)  # 재시작 처리 대기
                    
                    # 재시작 성공 확인 (여러 번 시도)
                    for attempt in range(3):
                        time.sleep(0.5)
                        new_state = self.client.get_game_state()
                        if new_state and not new_state.get('isGameOver', False) and new_state.get('score', 0) == 0:
                            print(f"🎮 새 게임 시작 확인 (시도 {attempt + 1}), 계속 플레이...")
                            break
                    else:
                        print(f"⚠️ 재시작 확인 실패")
                        break
                    continue
                else:
                    print(f"❌ 재시작 요청 실패")
                    break
            
            # 앙상블 액션 선택
            best_ai_name, best_action, all_evaluations, situation = self.ensemble_action(state)
            
            if best_ai_name is None or all_evaluations is None or situation is None:
                print("⚠️ 앙상블 평가 실패")
                continue
            
            # 선택된 AI 카운트 증가
            ai_selection_count[best_ai_name] += 1
            
            if self.client.send_action(best_action):
                step += 1
                if step % 10 == 0:
                    score = state.get('score', 0)
                    lines = state.get('lines', 0)
                    
                    print(f"🎯 스텝 {step}: 점수 {score}, 라인 {lines}")
                    print(f"   🏆 선택된 AI: {best_ai_name.upper()} (최종점수: {all_evaluations[best_ai_name]['final_score']:.3f})")
                    
                    # 상황 분석 표시
                    print(f"   📊 상황 분석: 위험도={situation['danger_level']:.1f}, 공간={situation['space_utilization']:.1f}, 형태={situation['shape_fitness']:.1f}")
                    
                    # 모든 AI의 평가 표시
                    print("   🤖 AI 평가 (Q-value × 우선순위 = 최종점수):")
                    for ai_name, eval_data in sorted(all_evaluations.items(), key=lambda x: x[1]['final_score'], reverse=True):
                        marker = "👑" if ai_name == best_ai_name else "  "
                        print(f"   {marker} {ai_name}: {eval_data['q_value']:.3f} × {eval_data['priority']:.1f} = {eval_data['final_score']:.3f} | {eval_data['action']}")
                    print("-" * 60)
            
            time.sleep(0.4)
        
        print(f"✅ 앙상블 플레이 완료! 총 {step} 액션")
        print("\n🏆 AI 선택 통계:")
        total_selections = sum(ai_selection_count.values())
        if total_selections > 0:
            for ai_name, count in sorted(ai_selection_count.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_selections) * 100
                print(f"   {ai_name}: {count}회 ({percentage:.1f}%)")
    
    def ai_battle(self, duration=300):
        """AI들이 번갈아가며 플레이하는 배틀"""
        print("⚔️  AI 배틀 모드!")
        print(f"🕐 총 시간: {duration}초")
        print("=" * 40)
        
        ai_list = list(self.ais.items())
        current_ai_idx = 0
        switch_interval = 60  # 60초마다 AI 교체
        
        start_time = time.time()
        last_switch = start_time
        
        while time.time() - start_time < duration:
            # AI 교체 시간 체크
            if time.time() - last_switch >= switch_interval:
                current_ai_idx = (current_ai_idx + 1) % len(ai_list)
                last_switch = time.time()
                print(f"\n🔄 AI 교체: {ai_list[current_ai_idx][0].upper()}")
            
            ai_name, ai = ai_list[current_ai_idx]
            
            state = self.client.get_game_state()
            if not state:
                time.sleep(0.5)
                continue
            
            # 게임 오버 처리 (강화된 감지)
            is_game_over = state.get('isGameOver', False)
            
            # 추가적인 게임 오버 감지 조건들
            if not is_game_over:
                board = state.get('board', [])
                if board:
                    # 상단 3줄에 블록이 있는지 확인
                    for row in range(3):
                        if any(board[row][col] is not None and board[row][col] != 0 for col in range(10)):
                            is_game_over = True
                            print(f"🚨 {ai_name} AI가 게임 오버 감지: 상단 영역 블록 발견 (행 {row})")
                            break
            
            if is_game_over:
                print(f"💀 {ai_name} 게임 오버 감지! 점수: {state.get('score', 0)}, 라인: {state.get('lines', 0)}")
                
                print(f"🔄 {ai_name} 즉시 재시작 시도...")
                if self.client.send_action('restart'):
                    print(f"✅ 재시작 요청 전송됨")
                    time.sleep(1.5)  # 재시작 처리 대기
                    
                    # 재시작 성공 확인 (여러 번 시도)
                    for attempt in range(3):
                        time.sleep(0.5)
                        new_state = self.client.get_game_state()
                        if new_state and not new_state.get('isGameOver', False) and new_state.get('score', 0) == 0:
                            print(f"🎮 {ai_name} 새 게임 시작 확인 (시도 {attempt + 1})")
                            break
                    else:
                        print(f"⚠️ {ai_name} 재시작 확인 실패")
                else:
                    print(f"❌ {ai_name} 재시작 요청 실패")
                continue
            
            # AI 액션
            old_epsilon = ai.epsilon
            ai.epsilon = 0
            action_idx = ai.get_action(state)
            action = ai.actions[action_idx]
            ai.epsilon = old_epsilon
            
            self.client.send_action(action)
            time.sleep(0.4)
        
        print(f"🏁 AI 배틀 완료!")

def main():
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch 설치 필요: pip install torch")
        return
    
    print("🧠 PyTorch DQN 다중 전문화 테트리스 AI")
    print("=" * 50)
    
    system = MultiAITetrisSystem()
    
    # 서버 연결 확인
    if not system.client.get_game_state():
        print("❌ 서버에 연결할 수 없습니다.")
        print("💡 'pnpm dev'로 서버를 시작하고 브라우저에서 'AI 모드 ON'을 클릭하세요.")
        return
    
    print("\n모드를 선택하세요:")
    print("1. 전체 AI 훈련")
    print("2. 특정 AI 플레이")
    print("3. AI 배틀 모드")
    print("4. 스마트 앙상블 AI 플레이 (우선순위 기반 생존 전략)")
    
    mode = input("선택 (1-4): ")
    
    if mode == "1":
        episodes = int(input("각 AI 훈련 에피소드 수 (기본 30): ") or "30")
        system.train_all(episodes=episodes)
    
    elif mode == "2":
        print("\n사용 가능한 AI:")
        for i, name in enumerate(system.ais.keys(), 1):
            print(f"{i}. {name}")
        
        ai_choice = input("AI 선택 (이름 또는 번호): ")
        
        # 번호로 선택한 경우
        if ai_choice.isdigit():
            ai_names = list(system.ais.keys())
            ai_idx = int(ai_choice) - 1
            if 0 <= ai_idx < len(ai_names):
                ai_choice = ai_names[ai_idx]
        
        duration = int(input("플레이 시간(초) (기본 120): ") or "120")
        system.play_with_ai(ai_choice, duration)
    
    elif mode == "3":
        duration = int(input("배틀 시간(초) (기본 300): ") or "300")
        system.ai_battle(duration)
    
    elif mode == "4":
        print("\n🎯 스마트 앙상블 AI 전략 설명:")
        print("  • 위험도 분석: 게임 오버 위험성에 따른 AI 우선순위 조정")
        print("  • 공간 활용: 넓은 공간이 있을 때 우선 활용, 위치 조정 장려") 
        print("  • 형태 맞춤: 현재 블록이 보드에 잘 맞는지 분석하여 회전/이동 결정")
        print("  • 생존 우선: 위험 상황에서 drop/rotate 신중 선택, 균등한 높이 유지 최우선")
        print("  • 상황별 전략: 안전/위험/극위험 상황에 따른 AI 역할 자동 조정")
        print("")
        duration = int(input("앙상블 플레이 시간(초) (기본 120): ") or "120")
        system.play_with_ensemble(duration)
    
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 