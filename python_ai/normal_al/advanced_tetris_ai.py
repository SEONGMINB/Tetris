import numpy as np
import time
from tetris_player import TetrisAI

class AdvancedTetrisAI(TetrisAI):
    def __init__(self, url="http://localhost:3000"):
        super().__init__(url)
        self.last_board = None
        self.consecutive_same_moves = 0
        self.last_action = None
        
    def evaluate_board(self, board):
        """보드 상태 평가 함수"""
        # 1. 전체 높이 (낮을수록 좋음)
        heights = self.get_column_heights(board)
        total_height = sum(heights)
        
        # 2. 구멍 개수 (적을수록 좋음)
        holes = self.count_holes(board)
        
        # 3. 높이 차이 (적을수록 좋음)
        height_variance = np.var(heights)
        
        # 4. 완성 가능한 라인 수 (많을수록 좋음)
        complete_lines = self.count_complete_lines(board)
        
        # 5. 깊은 구멍 penalty (깊을수록 나쁨)
        deep_holes = self.count_deep_holes(board)
        
        # 가중치를 적용한 점수 계산
        score = (
            -total_height * 0.510066 +
            -holes * 0.760666 +
            -height_variance * 0.35663 +
            complete_lines * 0.818 +
            -deep_holes * 1.2
        )
        
        return score
    
    def get_column_heights(self, board):
        """각 열의 높이를 계산"""
        heights = []
        rows, cols = board.shape
        
        for col in range(cols):
            height = 0
            for row in range(rows):
                if board[row][col] == 1:
                    height = rows - row
                    break
            heights.append(height)
        
        return heights
    
    def count_holes(self, board):
        """구멍 개수 계산"""
        holes = 0
        rows, cols = board.shape
        
        for col in range(cols):
            block_found = False
            for row in range(rows):
                if board[row][col] == 1:
                    block_found = True
                elif block_found and board[row][col] == 0:
                    holes += 1
        
        return holes
    
    def count_complete_lines(self, board):
        """완성된 라인 수 계산"""
        complete_lines = 0
        for row in board:
            if np.all(row == 1):
                complete_lines += 1
        return complete_lines
    
    def count_deep_holes(self, board):
        """깊은 구멍 계산 (위에 2개 이상 블록이 있는 구멍)"""
        deep_holes = 0
        rows, cols = board.shape
        
        for col in range(cols):
            blocks_above = 0
            for row in range(rows):
                if board[row][col] == 1:
                    blocks_above += 1
                elif board[row][col] == 0 and blocks_above >= 2:
                    deep_holes += 1
        
        return deep_holes
    
    def smart_ai_strategy(self, game_state):
        """향상된 AI 전략"""
        if game_state is None or game_state['game_over']:
            return None
        
        board = game_state['board']
        
        # 보드가 변하지 않으면 다른 액션 시도
        if self.last_board is not None and np.array_equal(board, self.last_board):
            self.consecutive_same_moves += 1
        else:
            self.consecutive_same_moves = 0
        
        self.last_board = board.copy()
        
        # 너무 많이 같은 상태면 강제로 다른 액션
        if self.consecutive_same_moves > 3:
            actions = ['rotate', 'left', 'right']
            if self.last_action in actions:
                actions.remove(self.last_action)
            import random
            action = random.choice(actions)
            self.last_action = action
            return action
        
        # 현재 보드 평가
        current_score = self.evaluate_board(board)
        
        # 가능한 액션들의 결과 시뮬레이션
        actions = ['left', 'right', 'rotate', 'down']
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            # 간단한 시뮬레이션 (실제로는 더 복잡한 예측 필요)
            simulated_score = current_score
            
            if action == 'down':
                # 아래로 이동은 일반적으로 안전
                simulated_score += 0.1
            elif action == 'rotate':
                # 회전은 상황에 따라 다름
                simulated_score += 0.05
            
            # 높이가 너무 높으면 아래로 이동 우선
            heights = self.get_column_heights(board)
            if max(heights) > 15 and action == 'down':
                simulated_score += 0.5
            
            # 구멍이 많으면 회전보다는 위치 조정
            holes = self.count_holes(board)
            if holes > 5 and action in ['left', 'right']:
                simulated_score += 0.3
            
            if simulated_score > best_score:
                best_score = simulated_score
                best_action = action
        
        self.last_action = best_action
        return best_action
    
    def play_smart(self, duration=120):
        """스마트 AI로 게임 플레이"""
        print("스마트 AI가 테트리스를 플레이합니다...")
        start_time = time.time()
        last_score = 0
        
        while time.time() - start_time < duration:
            game_state = self.get_game_state()
            
            if game_state is None:
                time.sleep(0.5)
                continue
                
            if game_state['game_over']:
                print(f"게임 오버! 최종 점수: {game_state['score']} - 자동 재시작...")
                self.send_action('restart')  # R키로 자동 재시작
                time.sleep(1)  # 재시작 후 잠시 대기
                print("🔄 게임 재시작됨, 계속 플레이...")
                continue
            
            # 점수 변화 출력
            if game_state['score'] != last_score:
                print(f"점수: {game_state['score']} (+{game_state['score'] - last_score}), "
                      f"레벨: {game_state['level']}, 라인: {game_state['lines']}")
                last_score = game_state['score']
            
            # 스마트 전략으로 액션 결정
            action = self.smart_ai_strategy(game_state)
            
            if action:
                self.send_action(action)
                
            time.sleep(0.2)  # 게임 속도 조절

def main():
    ai = AdvancedTetrisAI()
    
    try:
        print("고급 테트리스 AI를 시작합니다...")
        ai.start_game()
        ai.play_smart(duration=300)  # 5분간 플레이
        
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        ai.close()

if __name__ == "__main__":
    main() 