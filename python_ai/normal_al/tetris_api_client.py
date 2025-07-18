import requests
import time
import json
import numpy as np
from typing import Dict, List, Optional

class TetrisAPIClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/game"
        
    def get_game_state(self) -> Optional[Dict]:
        """현재 게임 상태를 가져옵니다."""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data.get('data')
            return None
        except Exception as e:
            print(f"게임 상태 조회 오류: {e}")
            return None
    
    def send_action(self, action: str) -> bool:
        """액션을 게임에 전송합니다."""
        try:
            response = requests.post(self.api_url, json={
                'type': 'action',
                'action': action
            })
            response.raise_for_status()
            
            data = response.json()
            return data.get('success', False)
        except Exception as e:
            print(f"액션 전송 오류: {e}")
            return False
    
    def wait_for_connection(self, timeout=30):
        """게임 서버 연결을 기다립니다."""
        print("게임 서버 연결을 확인 중...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.api_url, timeout=5)
                if response.status_code == 200:
                    print("✅ 게임 서버에 연결되었습니다!")
                    return True
            except:
                pass
            
            print(".", end="", flush=True)
            time.sleep(1)
        
        print(f"\n❌ {timeout}초 후에도 서버에 연결할 수 없습니다.")
        return False

class TetrisAI:
    def __init__(self, base_url="http://localhost:3000"):
        self.client = TetrisAPIClient(base_url)
        self.last_board = None
        self.consecutive_same_moves = 0
        self.last_action = None
        
    def evaluate_board(self, board: List[List]) -> float:
        """보드 상태를 평가합니다."""
        if not board:
            return 0
            
        board_array = np.array(board)
        
        # 1. 전체 높이
        heights = self.get_column_heights(board_array)
        total_height = sum(heights)
        
        # 2. 구멍 개수
        holes = self.count_holes(board_array)
        
        # 3. 높이 차이
        height_variance = float(np.var(heights)) if heights else 0.0
        
        # 4. 완성 라인
        complete_lines = self.count_complete_lines(board_array)
        
        # 5. 깊은 구멍
        deep_holes = self.count_deep_holes(board_array)
        
        # 가중치 적용 점수
        score = (
            -total_height * 0.510066 +
            -holes * 0.760666 +
            -height_variance * 0.35663 +
            complete_lines * 0.818 +
            -deep_holes * 1.2
        )
        
        return score
    
    def get_column_heights(self, board: np.ndarray) -> List[int]:
        """각 열의 높이를 계산합니다."""
        heights = []
        rows, cols = board.shape
        
        for col in range(cols):
            height = 0
            for row in range(rows):
                if board[row][col] != 0 and board[row][col] is not None:
                    height = rows - row
                    break
            heights.append(height)
        
        return heights
    
    def count_holes(self, board: np.ndarray) -> int:
        """구멍 개수를 계산합니다."""
        holes = 0
        rows, cols = board.shape
        
        for col in range(cols):
            block_found = False
            for row in range(rows):
                cell = board[row][col]
                if cell != 0 and cell is not None:
                    block_found = True
                elif block_found and (cell == 0 or cell is None):
                    holes += 1
        
        return holes
    
    def count_complete_lines(self, board: np.ndarray) -> int:
        """완성된 라인 수를 계산합니다."""
        complete_lines = 0
        for row in board:
            if all(cell != 0 and cell is not None for cell in row):
                complete_lines += 1
        return complete_lines
    
    def count_deep_holes(self, board: np.ndarray) -> int:
        """깊은 구멍을 계산합니다."""
        deep_holes = 0
        rows, cols = board.shape
        
        for col in range(cols):
            blocks_above = 0
            for row in range(rows):
                cell = board[row][col]
                if cell != 0 and cell is not None:
                    blocks_above += 1
                elif (cell == 0 or cell is None) and blocks_above >= 2:
                    deep_holes += 1
        
        return deep_holes
    
    def choose_action(self, game_state: Dict) -> Optional[str]:
        """게임 상태를 분석해서 최적의 액션을 선택합니다."""
        if not game_state or game_state.get('isGameOver') or game_state.get('isPaused'):
            return None
        
        board = game_state.get('board', [])
        if not board:
            return None
        
        # 보드가 변하지 않으면 다른 액션 시도
        current_board = np.array(board)
        if self.last_board is not None and np.array_equal(current_board, self.last_board):
            self.consecutive_same_moves += 1
        else:
            self.consecutive_same_moves = 0
        
        self.last_board = current_board.copy()
        
        # 같은 상태가 계속되면 강제 액션
        if self.consecutive_same_moves > 3:
            actions = ['rotate', 'left', 'right', 'down']
            if self.last_action in actions:
                actions.remove(self.last_action)
            import random
            action = random.choice(actions)
            self.last_action = action
            return action
        
        # 현재 보드 평가
        current_score = self.evaluate_board(board)
        
        # 간단한 휴리스틱 전략
        actions = ['left', 'right', 'rotate', 'down']
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            simulated_score = current_score
            
            # 기본 액션 점수
            if action == 'down':
                simulated_score += 0.1
            elif action == 'rotate':
                simulated_score += 0.05
            
            # 높이가 높으면 아래로 이동 우선
            heights = self.get_column_heights(current_board)
            if heights and max(heights) > 15 and action == 'down':
                simulated_score += 0.5
            
            # 구멍이 많으면 위치 조정 우선
            holes = self.count_holes(current_board)
            if holes > 5 and action in ['left', 'right']:
                simulated_score += 0.3
            
            if simulated_score > best_score:
                best_score = simulated_score
                best_action = action
        
        self.last_action = best_action
        return best_action
    
    def play(self, duration=120):
        """AI가 게임을 플레이합니다."""
        if not self.client.wait_for_connection():
            return
        
        print(f"\n🤖 AI가 {duration}초간 테트리스를 플레이합니다!")
        print("🎮 브라우저에서 'AI 모드 ON' 버튼을 클릭하세요!")
        print("=" * 50)
        
        start_time = time.time()
        last_score = 0
        action_count = 0
        
        while time.time() - start_time < duration:
            # 게임 상태 가져오기
            game_state = self.client.get_game_state()
            
            if not game_state:
                time.sleep(0.5)
                continue
            
            # 게임 오버 체크
            if game_state.get('isGameOver'):
                final_score = game_state.get('score', 0)
                print(f"\n🎯 게임 오버! 최종 점수: {final_score:,} - 자동 재시작...")
                print(f"📊 총 액션 수: {action_count}")
                # 자동으로 재시작 액션 전송
                if self.client.send_action('restart'):
                    print(f"🔄 게임 재시작됨, 계속 플레이...")
                    time.sleep(1)  # 재시작 후 잠시 대기
                    continue  # 플레이 계속 진행
                else:
                    print(f"❌ 재시작 실패")
                    break
            
            # 점수 변화 출력
            current_score = game_state.get('score', 0)
            if current_score != last_score:
                level = game_state.get('level', 0)
                lines = game_state.get('lines', 0)
                score_diff = current_score - last_score
                print(f"📈 점수: {current_score:,} (+{score_diff}), 레벨: {level}, 라인: {lines}")
                last_score = current_score
            
            # 액션 선택 및 전송
            action = self.choose_action(game_state)
            if action:
                success = self.client.send_action(action)
                if success:
                    action_count += 1
                    if action_count % 20 == 0:  # 20번째마다 출력
                        print(f"🎮 액션 #{action_count}: {action}")
            
            time.sleep(0.25)  # 250ms 간격으로 액션
        
        elapsed = time.time() - start_time
        print(f"\n✅ AI 플레이 완료!")
        print(f"⏱️  플레이 시간: {elapsed:.1f}초")
        print(f"🎯 최종 점수: {last_score:,}")
        print(f"🎮 총 액션 수: {action_count}")

def main():
    print("🎮 테트리스 API AI 클라이언트")
    print("=" * 40)
    
    # AI 인스턴스 생성
    ai = TetrisAI()
    
    try:
        # AI 플레이 시작
        ai.play(duration=180)  # 3분간 플레이
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 중단했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 