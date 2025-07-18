#!/usr/bin/env python3
"""
강화학습 기반 테트리스 AI
Deep Q-Network (DQN) 사용
"""
import requests
import time
import numpy as np
import random
from collections import deque
import json
import os

# TensorFlow/PyTorch가 없는 경우를 위한 간단한 Q-learning 구현
class QLearningAgent:
    def __init__(self, state_size=200, action_size=6, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.state_size = state_size  # 20x10 보드
        self.action_size = action_size  # left, right, down, rotate, drop
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-table 초기화 (실제로는 딕셔너리 사용)
        self.q_table = {}
        self.memory = deque(maxlen=2000)
        
        # 액션 매핑
        self.actions = ['left', 'right', 'down', 'rotate', 'drop', 'restart']
        
    def state_to_key(self, state):
        """게임 상태를 Q-table 키로 변환"""
        if not state or 'board' not in state:
            return "empty"
        
        board = state['board']
        # 보드를 간단한 특징으로 변환
        features = []
        
        # 1. 각 열의 높이
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(min(height, 10))  # 최대 10으로 제한
        
        # 2. 구멍 개수 (간단화)
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    block_found = True
                elif block_found and (board[row][col] is None or board[row][col] == 0):
                    holes += 1
        holes = min(holes, 20)  # 최대 20으로 제한
        
        # 3. 완성 가능한 라인
        complete_lines = 0
        for row in range(20):
            filled_cells = sum(1 for cell in board[row] if cell is not None and cell != 0)
            if filled_cells >= 8:  # 거의 완성된 라인
                complete_lines += 1
        
        # 특징을 문자열로 변환 (해시 키로 사용)
        key = f"h:{'-'.join(map(str, heights[:5]))}|holes:{holes}|lines:{complete_lines}"
        return key
    
    def get_action(self, state):
        """현재 상태에서 액션 선택 (ε-greedy)"""
        if np.random.random() <= self.epsilon:
            # 탐험: 랜덤 액션
            return random.choice(range(self.action_size))
        
        # 활용: Q-값이 가장 높은 액션
        state_key = self.state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        """경험을 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, prev_state, current_state):
        """보상 계산"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 점수 증가 보상
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.01
        
        # 라인 클리어 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        reward += lines_diff * 10
        
        # 게임 오버 패널티
        if current_state.get('isGameOver', False):
            reward -= 50
        
        # 높이 패널티
        if current_state.get('board'):
            board = current_state['board']
            max_height = 0
            for col in range(10):
                for row in range(20):
                    if board[row][col] is not None and board[row][col] != 0:
                        max_height = max(max_height, 20 - row)
                        break
            
            if max_height > 15:
                reward -= (max_height - 15) * 2
        
        return reward
    
    def replay(self, batch_size=32):
        """경험 재생을 통한 학습"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.state_to_key(state)
            next_state_key = self.state_to_key(next_state)
            
            # Q-table 초기화
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_size
            
            # Q-learning 업데이트
            target = reward
            if not done:
                target += self.discount_factor * max(self.q_table[next_state_key])
            
            # Q-값 업데이트
            self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        # epsilon 감소 (탐험 -> 활용)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Q-table 저장"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath):
        """Q-table 로드"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
                self.q_table = model_data.get('q_table', {})
                self.epsilon = model_data.get('epsilon', self.epsilon)
            print(f"✅ 모델 로드됨: {len(self.q_table)}개 상태")
        else:
            print("❌ 저장된 모델 없음, 새로 학습 시작")

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

class ReinforcementTetrisAI:
    def __init__(self):
        self.agent = QLearningAgent()
        self.client = TetrisAPIClient()
        self.model_path = "tetris_q_model.json"
        
        # 모델 로드
        self.agent.load_model(self.model_path)
    
    def train(self, episodes=100, max_steps=1000):
        """강화학습 훈련"""
        print("🧠 강화학습 훈련 시작!")
        print(f"📊 에피소드: {episodes}, 최대 스텝: {max_steps}")
        print("=" * 50)
        
        for episode in range(episodes):
            print(f"\n🎮 에피소드 {episode + 1}/{episodes}")
            
            # 게임 상태 초기화
            prev_state = self.client.get_game_state()
            if not prev_state:
                print("❌ 서버 연결 실패")
                continue
            
            total_reward = 0
            step = 0
            
            for step in range(max_steps):
                # 현재 상태 가져오기
                current_state = self.client.get_game_state()
                if not current_state:
                    break
                
                # 게임 오버 체크
                if current_state.get('isGameOver', False):
                    print(f"   💀 게임 오버! 스텝: {step} - 자동 재시작...")
                    # 자동으로 재시작 액션 전송
                    if self.client.send_action('restart'):
                        print(f"   🔄 게임 재시작됨")
                        time.sleep(1)  # 재시작 후 잠시 대기
                        continue  # 에피소드 계속 진행
                    else:
                        print(f"   ❌ 재시작 실패")
                        break
                
                # 액션 선택
                action_idx = self.agent.get_action(current_state)
                action = self.agent.actions[action_idx]
                
                # 액션 실행
                if not self.client.send_action(action):
                    continue
                
                time.sleep(0.3)  # 액션 간격 (더 안정적)
                
                # 다음 상태 가져오기
                next_state = self.client.get_game_state()
                if not next_state:
                    continue
                
                # 보상 계산
                reward = self.agent.calculate_reward(current_state, next_state)
                total_reward += reward
                
                # 경험 저장
                done = next_state.get('isGameOver', False)
                self.agent.remember(current_state, action_idx, reward, next_state, done)
                
                # 진행 상황 출력
                if step % 50 == 0:
                    score = next_state.get('score', 0)
                    lines = next_state.get('lines', 0)
                    print(f"   📈 스텝 {step}: 점수 {score}, 라인 {lines}, 보상 {reward:.2f}")
                
                prev_state = current_state
                
                if done:
                    break
            
            # 학습 수행
            self.agent.replay()
            
            # 에피소드 결과 출력
            final_score = current_state.get('score', 0) if current_state else 0
            final_lines = current_state.get('lines', 0) if current_state else 0
            
            print(f"   🎯 에피소드 완료!")
            print(f"   📊 최종 점수: {final_score}, 라인: {final_lines}")
            print(f"   🎁 총 보상: {total_reward:.2f}")
            print(f"   🔍 탐험률: {self.agent.epsilon:.3f}")
            print(f"   🧠 학습된 상태: {len(self.agent.q_table)}")
            
            # 주기적으로 모델 저장
            if (episode + 1) % 10 == 0:
                self.agent.save_model(self.model_path)
                print(f"   💾 모델 저장됨 (에피소드 {episode + 1})")
        
        # 최종 모델 저장
        self.agent.save_model(self.model_path)
        print(f"\n✅ 훈련 완료! 모델 저장됨: {self.model_path}")
    
    def play(self, duration=120):
        """학습된 AI로 플레이"""
        print("🤖 학습된 AI가 플레이합니다!")
        print(f"🕐 플레이 시간: {duration}초")
        print(f"🔍 탐험률: {self.agent.epsilon:.3f}")
        print("=" * 40)
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            state = self.client.get_game_state()
            if not state:
                time.sleep(0.5)
                continue
            
            if state.get('isGameOver', False):
                print(f"🎯 게임 오버! 점수: {state.get('score', 0)} - 자동 재시작...")
                # 자동으로 재시작 액션 전송
                if self.client.send_action('restart'):
                    print(f"🔄 게임 재시작됨, 계속 플레이...")
                    time.sleep(1)  # 재시작 후 잠시 대기
                    continue  # 플레이 계속 진행
                else:
                    print(f"❌ 재시작 실패")
                    break
            
            # 최적 액션 선택 (탐험 없이)
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0  # 순수 활용 모드
            
            action_idx = self.agent.get_action(state)
            action = self.agent.actions[action_idx]
            
            self.agent.epsilon = old_epsilon
            
            # 액션 실행
            if self.client.send_action(action):
                step += 1
                if step % 20 == 0:
                    score = state.get('score', 0)
                    lines = state.get('lines', 0)
                    print(f"🎮 스텝 {step}: 점수 {score}, 라인 {lines}, 액션 {action}")
            
            time.sleep(0.4)  # 플레이 모드에서 더 안정적인 간격
        
        print(f"✅ 플레이 완료! 총 {step} 액션")

def main():
    print("🧠 강화학습 테트리스 AI")
    print("=" * 40)
    
    ai = ReinforcementTetrisAI()
    
    # 서버 연결 확인
    if not ai.client.get_game_state():
        print("❌ 서버에 연결할 수 없습니다.")
        print("💡 'pnpm dev'로 서버를 시작하고 브라우저에서 'AI 모드 ON'을 클릭하세요.")
        return
    
    mode = input("\n모드를 선택하세요:\n1. 훈련 (train)\n2. 플레이 (play)\n선택 (1 또는 2): ")
    
    if mode == "1":
        episodes = int(input("훈련 에피소드 수 (기본 50): ") or "50")
        ai.train(episodes=episodes)
    elif mode == "2":
        duration = int(input("플레이 시간(초) (기본 120): ") or "120")
        ai.play(duration=duration)
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 