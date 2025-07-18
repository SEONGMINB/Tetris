#!/usr/bin/env python3
"""
단순한 보상 시스템 강화학습 테트리스 AI
- 높이 감소: +1점/칸
- 라인 클리어: +20점/라인
"""
import requests
import time
import numpy as np
import random
from collections import deque
import json
import os

class SimpleRewardAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.9, epsilon=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        self.q_table = {}
        self.memory = deque(maxlen=3000)
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        
    def get_max_height(self, board):
        """보드의 최대 높이 계산"""
        max_height = 0
        for col in range(10):
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    max_height = max(max_height, height)
                    break
        return max_height
    
    def get_piece_placement_rows(self, prev_board, current_board):
        """새로 배치된 피스가 놓인 행들을 찾기"""
        if not prev_board or not current_board:
            return []
        
        placed_rows = []
        for row in range(20):
            for col in range(10):
                prev_cell = prev_board[row][col]
                current_cell = current_board[row][col]
                
                # 이전에 없던 블럭이 새로 생겼으면
                if (prev_cell is None or prev_cell == 0) and (current_cell is not None and current_cell != 0):
                    if row not in placed_rows:
                        placed_rows.append(row)
        
        return placed_rows
    
    def count_holes(self, board):
        """보드에서 구멍(위에 블록이 있는데 아래 빈 공간) 개수 계산"""
        if not board:
            return 0
        
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    block_found = True
                elif block_found and (board[row][col] is None or board[row][col] == 0):
                    holes += 1
        
        return holes
    
    def calculate_hole_filling_reward(self, prev_state, current_state):
        """구멍 메우기 보상 계산"""
        if not prev_state.get('board') or not current_state.get('board'):
            return 0
        
        prev_holes = self.count_holes(prev_state['board'])
        current_holes = self.count_holes(current_state['board'])
        
        holes_filled = prev_holes - current_holes
        
        if holes_filled > 0:
            # 구멍을 메울 때마다 50점씩 추가
            hole_reward = holes_filled * 50
            print(f"   🧩 구멍 메우기! {holes_filled}개 구멍 = +{hole_reward}점")
            return hole_reward
        
        return 0
    
    def get_column_heights(self, board):
        """각 열의 높이를 계산"""
        if not board:
            return [0] * 10
        
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = 20 - row
                    break
            heights.append(height)
        return heights
    
    def calculate_tetris_setup_reward(self, current_state):
        """테트리스 셋업 보상 - I-piece로 클리어할 수 있는 모양 만들기"""
        if not current_state.get('board'):
            return 0
        
        board = current_state['board']
        heights = self.get_column_heights(board)
        
        # 테트리스 셋업 감지: 한 열만 낮고 나머지 열들이 거의 같은 높이
        setup_reward = 0
        
        for target_col in range(10):
            target_height = heights[target_col]
            other_heights = [h for i, h in enumerate(heights) if i != target_col]
            
            if not other_heights:
                continue
            
            # 다른 열들의 평균 높이
            avg_other_height = sum(other_heights) / len(other_heights)
            
            # 조건: 타겟 열이 다른 열들보다 3-4칸 낮고, 다른 열들이 비슷한 높이
            height_diff = avg_other_height - target_height
            
            if 3 <= height_diff <= 4:
                # 다른 열들이 얼마나 고르게 쌓였는지 확인
                height_variance = sum((h - avg_other_height) ** 2 for h in other_heights) / len(other_heights)
                
                if height_variance <= 1.0:  # 높이 차이가 적을 때
                    setup_points = int(height_diff * 30)  # 3칸 차이 = 90점, 4칸 차이 = 120점
                    setup_reward += setup_points
                    print(f"   🎯 테트리스 셋업! {target_col+1}번 열 대기 = +{setup_points}점")
        
        return setup_reward
    
    def calculate_flatness_reward(self, prev_state, current_state):
        """평탄성 보상 - 네모진 모양으로 쌓이면 보상"""
        if not prev_state.get('board') or not current_state.get('board'):
            return 0
        
        prev_heights = self.get_column_heights(prev_state['board'])
        current_heights = self.get_column_heights(current_state['board'])
        
        # 이전과 현재의 높이 분산 계산
        def calculate_height_variance(heights):
            if not heights:
                return 0
            avg_height = sum(heights) / len(heights)
            return sum((h - avg_height) ** 2 for h in heights) / len(heights)
        
        prev_variance = calculate_height_variance(prev_heights)
        current_variance = calculate_height_variance(current_heights)
        
        # 분산이 줄어들었으면 (더 평탄해졌으면) 보상
        variance_improvement = prev_variance - current_variance
        
        if variance_improvement > 0:
            flatness_reward = int(variance_improvement * 20)  # 분산 개선 정도에 비례
            print(f"   📏 평탄화! 높이 균등화 = +{flatness_reward}점")
            return flatness_reward
        
        return 0

    def calculate_depth_reward(self, prev_state, current_state):
        """깊이 배치 보상 계산 - 아래쪽에 놓을수록 더 높은 점수"""
        if not prev_state.get('board') or not current_state.get('board'):
            return 0
        
        prev_board = prev_state['board']
        current_board = current_state['board']
        
        # 새로 배치된 피스의 행들 찾기
        placed_rows = self.get_piece_placement_rows(prev_board, current_board)
        
        if not placed_rows:
            return 0
        
        total_depth_reward = 0
        
        for row in placed_rows:
            # 깊이 보상: 맨 아래(row 19) = 20점, 위로 갈수록 감소
            depth_score = 20 - row  # row 19 = 1점, row 18 = 2점, ..., row 0 = 20점
            
            # 추가 보너스: 정말 깊은 곳 (15행 이하)에 놓으면 보너스
            if row >= 15:  # 아래쪽 5줄
                depth_bonus = (row - 14) * 2  # 15행=2점, 16행=4점, ..., 19행=10점
                depth_score += depth_bonus
                total_depth_reward += depth_score
                print(f"   🏔️ 깊이 배치! {20-row}번째 줄 = +{depth_score}점 (깊이보너스 +{depth_bonus})")
            else:
                total_depth_reward += depth_score
                print(f"   ⬇️ 배치! {20-row}번째 줄 = +{depth_score}점")
        
        return total_depth_reward
    
    def state_to_key(self, state):
        """게임 상태를 간단한 키로 변환"""
        if not state or 'board' not in state:
            return "empty"
        
        board = state['board']
        max_height = self.get_max_height(board)
        
        # 각 열의 높이 (최대 5칸까지만 고려)
        heights = []
        for col in range(10):
            height = 0
            for row in range(20):
                if board[row][col] is not None and board[row][col] != 0:
                    height = min(20 - row, 10)  # 최대 10으로 제한
                    break
            heights.append(height)
        
        # 높이를 그룹화 (상태 공간 축소)
        height_groups = [min(h // 2, 5) for h in heights]  # 0-5 그룹
        max_height_group = min(max_height // 3, 6)  # 0-6 그룹
        
        # 현재 피스 타입
        piece_type = state.get('currentPiece', {}).get('type', 'None')
        
        key = f"max_h:{max_height_group}|h:{'-'.join(map(str, height_groups[:5]))}|piece:{piece_type}"
        return key
    
    def get_action(self, state):
        """액션 선택 (ε-greedy)"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(len(self.actions)))
        
        state_key = self.state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.actions)
        
        return np.argmax(self.q_table[state_key])
    
    def calculate_reward(self, prev_state, current_state):
        """간단한 보상 계산"""
        if not prev_state or not current_state:
            return 0
        
        reward = 0
        
        # 1. 개선된 라인 클리어 보상 시스템
        prev_lines = prev_state.get('lines', 0)
        current_lines = current_state.get('lines', 0)
        lines_cleared = current_lines - prev_lines
        
        if lines_cleared > 0:
            # 새로운 보상 시스템: 더 많은 줄을 한번에 클리어할수록 더 큰 보상
            if lines_cleared == 1:
                line_reward = 100
                emoji = "🟡"
                name = "싱글"
            elif lines_cleared == 2:
                line_reward = 200
                emoji = "🟠"
                name = "더블"
            elif lines_cleared == 3:
                line_reward = 300
                emoji = "🔴"
                name = "트리플"
            elif lines_cleared == 4:
                line_reward = 500  # 테트리스는 특별히 더 큰 보상!
                emoji = "💎"
                name = "테트리스"
            else:
                line_reward = lines_cleared * 100  # 혹시 5줄 이상이면
                emoji = "🌟"
                name = "슈퍼"
            
            reward += line_reward
            print(f"   {emoji} {name}! {lines_cleared}줄 클리어 = +{line_reward}점")
        
        # 2. 강화된 높이 관리 보상
        if current_state.get('board'):
            current_max_height = self.get_max_height(current_state['board'])
            prev_max_height = self.get_max_height(prev_state['board']) if prev_state.get('board') else 0
            
            # 기존 높이 감소 보상 (강화)
            height_decrease = prev_max_height - current_max_height
            if height_decrease > 0:
                height_reward = height_decrease * 3  # 기존 1점에서 3점으로 증가
                reward += height_reward
                print(f"   📉 높이 감소! {height_decrease}칸 = +{height_reward}점")
            
            # 새로운 깊이 배치 보상 시스템
            depth_reward = self.calculate_depth_reward(prev_state, current_state)
            if depth_reward > 0:
                reward += depth_reward
            
            # 구멍 메우기 보상 시스템
            hole_filling_reward = self.calculate_hole_filling_reward(prev_state, current_state)
            if hole_filling_reward > 0:
                reward += hole_filling_reward
            
            # 테트리스 셋업 보상 시스템 (I-piece로 클리어 가능한 형태)
            tetris_setup_reward = self.calculate_tetris_setup_reward(current_state)
            if tetris_setup_reward > 0:
                reward += tetris_setup_reward
            
            # 평탄성 보상 시스템 (네모진 모양으로 균등하게 쌓기)
            flatness_reward = self.calculate_flatness_reward(prev_state, current_state)
            if flatness_reward > 0:
                reward += flatness_reward
        
        # 3. 게임 오버 시 작은 패널티
        if current_state.get('isGameOver', False):
            reward -= 5
            print(f"   💀 게임 오버 = -5점")
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """경험을 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """경험 재생을 통한 학습"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.state_to_key(state)
            next_state_key = self.state_to_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * len(self.actions)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * len(self.actions)
            
            target = reward
            if not done:
                target += self.discount_factor * max(self.q_table[next_state_key])
            
            # Q-값 업데이트
            old_value = self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * (target - old_value)
        
        # epsilon 감소 (탐험 -> 활용)
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

class SimpleRewardTetrisAI:
    def __init__(self):
        self.agent = SimpleRewardAgent()
        self.client = TetrisAPIClient()
        self.model_path = "simple_reward_tetris_model.json"
        
        # 모델 로드
        self.agent.load_model(self.model_path)
    
    def wait_for_game_restart(self):
        """자동으로 게임을 재시작하고 시작을 기다리는 함수"""
        print("   ⏳ 게임 오버! AI가 자동으로 재시작합니다...")
        
        # 자동으로 restart 액션 보내기
        restart_success = self.client.send_action('restart')
        if restart_success:
            print("   🔄 재시작 명령 전송됨! 새 게임 시작 대기 중...", end="", flush=True)
        else:
            print("   ❌ 재시작 명령 전송 실패! 수동 재시작을 기다립니다...")
            print("   🔄 브라우저에서 R 키를 눌러주세요...", end="", flush=True)
        
        start_wait_time = time.time()
        dots = 0
        
        while True:
            time.sleep(1)  # 1초마다 체크 (더 빠르게)
            state = self.client.get_game_state()
            
            if state and not state.get('isGameOver', False):
                # 게임이 재시작되었는지 확인 (점수와 라인이 리셋되었는지)
                if state.get('score', 0) == 0 and state.get('lines', 0) == 0:
                    print("\n   ✅ 새 게임 시작됨!")
                    return state
            
            # 대기 중 표시
            dots = (dots + 1) % 4
            print("." * dots + " " * (3 - dots), end="\r", flush=True)
            
            # 타임아웃 (15초로 단축)
            if time.time() - start_wait_time > 15:
                print("\n   ⏰ 15초 타임아웃. 다음 에피소드로 건너뜁니다.")
                return None

    def train(self, episodes=500, max_steps=200):
        """강화학습 훈련"""
        print("🧠 단순 보상 시스템 강화학습 훈련!")
        print("=" * 50)
        print("🎁 깊이 중심 보상 시스템:")
        print("   • 높이 감소: +3점/칸 (강화)")
        print("   • 깊이 배치: 1줄=20점, 2줄=19점, ..., 20줄=1점")
        print("   • 깊이 보너스: 아래 5줄에 배치시 추가 2-10점")
        print("   • 1줄 클리어: +100점 🟡")
        print("   • 2줄 클리어: +200점 🟠")
        print("   • 3줄 클리어: +300점 🔴")
        print("   • 4줄 클리어: +500점 💎 (테트리스!)")
        print("   • 게임 오버: -5점")
        print("💡 게임 오버 시 자동으로 재시작 대기합니다!")
        print("=" * 50)
        
        for episode in range(episodes):
            print(f"\n🎮 에피소드 {episode + 1}/{episodes}")
            
            # 게임 상태 가져오기
            prev_state = self.client.get_game_state()
            if not prev_state:
                print("❌ 서버 연결 실패")
                continue
            
            # 게임이 이미 오버면 재시작 대기
            if prev_state.get('isGameOver', False):
                prev_state = self.wait_for_game_restart()
                if not prev_state:
                    continue
            
            total_reward = 0
            step = 0
            last_reward_step = 0
            
            print(f"   🚀 시작! 현재 점수: {prev_state.get('score', 0)}, 라인: {prev_state.get('lines', 0)}")
            
            for step in range(max_steps):
                # 현재 상태 가져오기
                current_state = self.client.get_game_state()
                if not current_state:
                    print("   ❌ 상태 가져오기 실패")
                    break
                
                # 게임 오버 체크
                if current_state.get('isGameOver', False):
                    print(f"   💀 게임 오버! 스텝: {step}")
                    # 게임 오버도 학습에 포함
                    reward = self.agent.calculate_reward(prev_state, current_state)
                    if reward != 0:
                        total_reward += reward
                    
                    # 학습에 게임 오버 상태 저장
                    self.agent.remember(prev_state, action_idx, reward, current_state, True)
                    break
                
                # 액션 선택
                action_idx = self.agent.get_action(current_state)
                action = self.agent.actions[action_idx]
                
                # 액션 실행
                if not self.client.send_action(action):
                    continue
                
                time.sleep(0.1)  # 적당한 간격
                
                # 다음 상태 가져오기
                next_state = self.client.get_game_state()
                if not next_state:
                    continue
                
                # 보상 계산
                reward = self.agent.calculate_reward(current_state, next_state)
                if reward != 0:
                    total_reward += reward
                    last_reward_step = step
                    print(f"      ⭐ 스텝 {step}: 보상 {reward:+.1f} (누적: {total_reward:.1f})")
                
                # 경험 저장
                done = next_state.get('isGameOver', False)
                self.agent.remember(current_state, action_idx, reward, next_state, done)
                
                # 진행 상황 출력
                if step % 40 == 0 and step > 0:
                    score = next_state.get('score', 0)
                    lines = next_state.get('lines', 0)
                    print(f"   📊 스텝 {step}: 점수 {score}, 라인 {lines}")
                
                prev_state = current_state
                
                if done:
                    break
            
            # 학습 수행
            self.agent.replay()
            
            # 에피소드 결과
            final_state = self.client.get_game_state()
            if final_state:
                final_score = final_state.get('score', 0)
                final_lines = final_state.get('lines', 0)
                print(f"   🎯 에피소드 완료!")
                print(f"   📊 최종 점수: {final_score}, 라인: {final_lines}")
                print(f"   🎁 총 보상: {total_reward:.1f}")
                print(f"   🔍 탐험률: {self.agent.epsilon:.3f}")
                print(f"   🧠 학습된 상태: {len(self.agent.q_table)}")
                print(f"   ⏰ 마지막 보상: 스텝 {last_reward_step}")
            
            # 주기적으로 모델 저장
            if (episode + 1) % 5 == 0:
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
        last_score = 0
        last_lines = 0
        
        while time.time() - start_time < duration:
            state = self.client.get_game_state()
            if not state:
                time.sleep(0.5)
                continue
            
            if state.get('isGameOver', False):
                print(f"🎯 게임 오버! 점수: {state.get('score', 0)}")
                print("🔄 게임 재시작을 기다리는 중...")
                
                # 재시작 대기
                restart_state = self.wait_for_game_restart()
                if restart_state:
                    print("✅ 게임 재시작됨! 플레이 계속...")
                    last_score = 0
                    last_lines = 0
                    continue
                else:
                    break
            
            # 학습된 정책으로 액션 선택 (탐험 없이)
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.05  # 약간의 랜덤성 유지
            
            action_idx = self.agent.get_action(state)
            action = self.agent.actions[action_idx]
            
            self.agent.epsilon = old_epsilon
            
            # 액션 실행
            if self.client.send_action(action):
                step += 1
                
                current_score = state.get('score', 0)
                current_lines = state.get('lines', 0)
                
                # 점수나 라인이 변했을 때 출력
                if current_score != last_score or current_lines != last_lines:
                    score_diff = current_score - last_score
                    lines_diff = current_lines - last_lines
                    print(f"🎮 스텝 {step}: 점수 {current_score} (+{score_diff}), 라인 {current_lines} (+{lines_diff}), 액션 {action}")
                    last_score = current_score
                    last_lines = current_lines
                elif step % 50 == 0:  # 50스텝마다 상태 출력
                    print(f"🎮 스텝 {step}: 점수 {current_score}, 라인 {current_lines}, 액션 {action}")
            
            time.sleep(0.15)
        
        print(f"✅ 플레이 완료! 총 {step} 액션")

def main():
    print("🧠 단순 보상 시스템 강화학습 테트리스 AI")
    print("=" * 50)
    
    ai = SimpleRewardTetrisAI()
    
    # 서버 연결 확인
    state = ai.client.get_game_state()
    if not state:
        print("❌ 서버에 연결할 수 없습니다.")
        print("💡 해결책:")
        print("   1. 터미널에서 'pnpm dev' 실행")
        print("   2. 브라우저에서 http://localhost:3000 접속")
        print("   3. '🤖 AI 모드 ON' 버튼 클릭")
        return
    
    # 현재 게임 상태 표시
    print(f"🎮 현재 게임 상태:")
    print(f"   점수: {state.get('score', 0)}")
    print(f"   라인: {state.get('lines', 0)}")
    print(f"   게임오버: {state.get('isGameOver', False)}")
    
    mode = input("\n모드를 선택하세요:\n1. 훈련 (train)\n2. 플레이 (play)\n선택 (1 또는 2): ")
    
    if mode == "1":
        episodes = int(input("훈련 에피소드 수 (기본 500): ") or "500")
        ai.train(episodes=episodes)
    elif mode == "2":
        duration = int(input("플레이 시간(초) (기본 120): ") or "120")
        ai.play(duration=duration)
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 