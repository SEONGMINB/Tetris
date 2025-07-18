#!/usr/bin/env python3
"""
개선된 보상 시스템을 가진 강화학습 테트리스 AI
"""
import requests
import time
import numpy as np
import random
from collections import deque
import json
import os

class ImprovedQLearningAgent:
    def __init__(self, state_size=200, action_size=5, learning_rate=0.15, discount_factor=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_table = {}
        self.memory = deque(maxlen=5000)  # 메모리 증가
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        
        # 이전 상태 추적
        self.prev_holes = 0
        self.prev_max_height = 0
        self.prev_bumpiness = 0
        
    def state_to_key(self, state):
        """게임 상태를 더 정교한 특징으로 변환"""
        if not state or 'board' not in state:
            return "empty"
        
        board = state['board']
        
        # 1. 각 열의 높이
        heights = self.get_column_heights(board)
        
        # 2. 구멍 개수
        holes = self.count_holes(board)
        
        # 3. 높이 차이 (bumpiness)
        bumpiness = self.calculate_bumpiness(heights)
        
        # 4. 완성 가능한 라인
        almost_complete = self.count_almost_complete_lines(board)
        
        # 5. 최대 높이
        max_height = max(heights) if heights else 0
        
        # 특징을 간단하게 그룹화 (상태 공간 축소)
        height_group = min(max_height // 3, 6)  # 0-6 그룹
        holes_group = min(holes // 2, 5)  # 0-5 그룹
        bump_group = min(bumpiness // 3, 4)  # 0-4 그룹
        
        key = f"h:{height_group}|holes:{holes_group}|bump:{bump_group}|lines:{almost_complete}"
        return key
    
    def get_column_heights(self, board):
        """각 열의 높이 계산"""
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def count_holes(self, board):
        """구멍 개수 계산"""
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                cell = board[row][col]
                if cell is not None and cell != 0:
                    block_found = True
                elif block_found and (cell is None or cell == 0):
                    holes += 1
        return holes
    
    def calculate_bumpiness(self, heights):
        """높이 차이의 합 계산"""
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def count_almost_complete_lines(self, board):
        """거의 완성된 라인 수 계산"""
        almost_complete = 0
        for row in range(20):
            filled_cells = sum(1 for cell in board[row] if cell is not None and cell != 0)
            if filled_cells >= 8:  # 8칸 이상 채워진 라인
                almost_complete += 1
        return almost_complete
    
    def get_action(self, state):
        """액션 선택 (ε-greedy)"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_key = self.state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        
        return np.argmax(self.q_table[state_key])
    
    def calculate_reward(self, prev_state, current_state, action):
        """개선된 보상 계산"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 1. 생존 보상 (매 스텝마다)
        if not current_state.get('isGameOver', False):
            reward += 1.0  # 생존 자체에 보상
        
        # 2. 점수 증가 보상 (증폭)
        score_diff = current_state.get('score', 0) - prev_state.get('score', 0)
        reward += score_diff * 0.1  # 기존 0.01에서 0.1로 증가
        
        # 3. 라인 클리어 대폭 보상
        lines_diff = current_state.get('lines', 0) - prev_state.get('lines', 0)
        if lines_diff > 0:
            reward += lines_diff * 25  # 기존 10에서 25로 증가
            if lines_diff >= 4:  # 테트리스!
                reward += 50  # 보너스
        
        # 4. 보드 상태 기반 보상/패널티
        if current_state.get('board'):
            board = current_state['board']
            heights = self.get_column_heights(board)
            holes = self.count_holes(board)
            max_height = max(heights) if heights else 0
            bumpiness = self.calculate_bumpiness(heights)
            
            # 구멍 생성 패널티
            hole_diff = holes - self.prev_holes
            if hole_diff > 0:
                reward -= hole_diff * 8  # 새로운 구멍마다 -8점
            
            # 높이 관리
            height_diff = max_height - self.prev_max_height
            if height_diff > 0:
                reward -= height_diff * 2  # 높이 증가 패널티
            elif height_diff < 0:
                reward += abs(height_diff) * 1  # 높이 감소 보상
            
            # 높이가 너무 높으면 강한 패널티
            if max_height > 16:
                reward -= (max_height - 16) * 5
            
            # 균등한 높이 보상
            bumpiness_diff = bumpiness - self.prev_bumpiness
            if bumpiness_diff < 0:
                reward += 2  # 더 균등해지면 보상
            elif bumpiness_diff > 0:
                reward -= 1  # 더 불균등해지면 패널티
            
            # 상태 업데이트
            self.prev_holes = holes
            self.prev_max_height = max_height
            self.prev_bumpiness = bumpiness
        
        # 5. 특정 액션에 대한 피드백
        if action == 'drop':
            reward += 0.5  # 하드 드롭 약간 선호
        elif action == 'down':
            reward += 0.2  # 빠른 낙하 약간 선호
        
        # 6. 게임 오버 패널티 (감소)
        if current_state.get('isGameOver', False):
            reward -= 30  # 기존 50에서 30으로 감소
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """경험을 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):  # 배치 크기 증가
        """경험 재생을 통한 학습"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.state_to_key(state)
            next_state_key = self.state_to_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_size
            
            target = reward
            if not done:
                target += self.discount_factor * max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        # epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """모델 저장"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath):
        """모델 로드"""
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

class ImprovedTetrisAI:
    def __init__(self):
        self.agent = ImprovedQLearningAgent()
        self.client = TetrisAPIClient()
        self.model_path = "improved_tetris_q_model.json"
        
        # 모델 로드
        self.agent.load_model(self.model_path)
    
    def train(self, episodes=50, max_steps=500):  # 더 짧은 에피소드
        """강화학습 훈련"""
        print("🧠 개선된 강화학습 훈련 시작!")
        print(f"📊 에피소드: {episodes}, 최대 스텝: {max_steps}")
        print("🎁 새로운 보상 시스템:")
        print("   • 생존 보상: +1.0/스텝")
        print("   • 라인 클리어: +25/라인 (테트리스 +50 보너스)")
        print("   • 구멍 생성: -8/구멍")
        print("   • 높이 관리: ±1-5점")
        print("=" * 60)
        
        for episode in range(episodes):
            print(f"\n🎮 에피소드 {episode + 1}/{episodes}")
            
            # 게임 상태 초기화
            prev_state = self.client.get_game_state()
            if not prev_state:
                print("❌ 서버 연결 실패")
                continue
            
            # 게임이 이미 오버면 넘어가기
            if prev_state.get('isGameOver', False):
                print("   ⚠️ 게임이 이미 오버 상태입니다. 브라우저에서 재시작하세요.")
                continue
            
            # 상태 초기화
            if prev_state.get('board'):
                board = prev_state['board']
                heights = self.agent.get_column_heights(board)
                self.agent.prev_holes = self.agent.count_holes(board)
                self.agent.prev_max_height = max(heights) if heights else 0
                self.agent.prev_bumpiness = self.agent.calculate_bumpiness(heights)
            
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
                
                time.sleep(0.15)  # 약간 더 긴 간격
                
                # 다음 상태 가져오기
                next_state = self.client.get_game_state()
                if not next_state:
                    continue
                
                # 보상 계산
                reward = self.agent.calculate_reward(current_state, next_state, action)
                total_reward += reward
                
                # 경험 저장
                done = next_state.get('isGameOver', False)
                self.agent.remember(current_state, action_idx, reward, next_state, done)
                
                # 진행 상황 출력
                if step % 30 == 0 and step > 0:  # 더 자주 출력
                    score = next_state.get('score', 0)
                    lines = next_state.get('lines', 0)
                    print(f"   📈 스텝 {step}: 점수 {score}, 라인 {lines}, 보상 {reward:.1f} (누적: {total_reward:.1f})")
                
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
            print(f"   🎁 총 보상: {total_reward:.1f}")
            print(f"   🔍 탐험률: {self.agent.epsilon:.3f}")
            print(f"   🧠 학습된 상태: {len(self.agent.q_table)}")
            
            # 주기적으로 모델 저장
            if (episode + 1) % 5 == 0:  # 더 자주 저장
                self.agent.save_model(self.model_path)
                print(f"   💾 모델 저장됨 (에피소드 {episode + 1})")
        
        # 최종 모델 저장
        self.agent.save_model(self.model_path)
        print(f"\n✅ 훈련 완료! 모델 저장됨: {self.model_path}")
    
    def play(self, duration=120):
        """학습된 AI로 플레이"""
        print("🤖 개선된 AI가 플레이합니다!")
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
            self.agent.epsilon = 0
            
            action_idx = self.agent.get_action(state)
            action = self.agent.actions[action_idx]
            
            self.agent.epsilon = old_epsilon
            
            # 액션 실행
            if self.client.send_action(action):
                step += 1
                if step % 15 == 0:
                    score = state.get('score', 0)
                    lines = state.get('lines', 0)
                    print(f"🎮 스텝 {step}: 점수 {score}, 라인 {lines}, 액션 {action}")
            
            time.sleep(0.2)
        
        print(f"✅ 플레이 완료! 총 {step} 액션")

def main():
    print("🧠 개선된 강화학습 테트리스 AI")
    print("=" * 40)
    
    ai = ImprovedTetrisAI()
    
    # 서버 연결 확인
    if not ai.client.get_game_state():
        print("❌ 서버에 연결할 수 없습니다.")
        print("💡 'pnpm dev'로 서버를 시작하고 브라우저에서 'AI 모드 ON'을 클릭하세요.")
        return
    
    mode = input("\n모드를 선택하세요:\n1. 훈련 (train)\n2. 플레이 (play)\n선택 (1 또는 2): ")
    
    if mode == "1":
        episodes = int(input("훈련 에피소드 수 (기본 30): ") or "30")
        ai.train(episodes=episodes)
    elif mode == "2":
        duration = int(input("플레이 시간(초) (기본 120): ") or "120")
        ai.play(duration=duration)
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 