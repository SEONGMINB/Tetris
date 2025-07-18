import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import cv2
from PIL import Image
import io
import base64

class TetrisAI:
    def __init__(self, url="http://localhost:3000"):
        self.url = url
        self.driver = None
        self.setup_driver()
        
    def setup_driver(self):
        """Chrome 웹드라이버 설정"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1200,800")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def start_game(self):
        """게임 시작"""
        self.driver.get(self.url)
        time.sleep(3)  # 게임 로딩 대기
        print("테트리스 게임이 시작되었습니다!")
        
    def get_game_state(self):
        """현재 게임 상태 읽기"""
        try:
            # 점수 읽기
            score_element = self.driver.find_element(By.CLASS_NAME, "score")
            score_text = score_element.text
            score = int(score_text.split(":")[1].strip().replace(",", ""))
            
            # 레벨 읽기
            level_elements = self.driver.find_elements(By.CLASS_NAME, "level")
            level = int(level_elements[0].text.split(":")[1].strip()) if level_elements else 0
            lines = int(level_elements[1].text.split(":")[1].strip()) if len(level_elements) > 1 else 0
            
            # 게임오버 체크
            game_over = len(self.driver.find_elements(By.CLASS_NAME, "game-over")) > 0
            
            # 보드 상태 읽기 (간단한 방식)
            board = self.get_board_state()
            
            return {
                'score': score,
                'level': level,
                'lines': lines,
                'game_over': game_over,
                'board': board
            }
        except Exception as e:
            print(f"게임 상태 읽기 오류: {e}")
            return None
    
    def get_board_state(self):
        """보드 상태를 2D 배열로 반환"""
        try:
            # 테트리스 셀들 가져오기
            cells = self.driver.find_elements(By.CLASS_NAME, "tetris-cell")
            
            # 20x10 보드로 변환
            board = []
            for row in range(20):
                board_row = []
                for col in range(10):
                    cell_index = row * 10 + col
                    if cell_index < len(cells):
                        cell = cells[cell_index]
                        # 채워진 셀인지 확인 (class에 'filled'가 있는지)
                        is_filled = 'filled' in cell.get_attribute('class')
                        board_row.append(1 if is_filled else 0)
                    else:
                        board_row.append(0)
                board.append(board_row)
            
            return np.array(board)
        except Exception as e:
            print(f"보드 상태 읽기 오류: {e}")
            return np.zeros((20, 10))
    
    def send_action(self, action):
        """액션 실행"""
        body = self.driver.find_element(By.TAG_NAME, "body")
        
        action_map = {
            'left': Keys.ARROW_LEFT,
            'right': Keys.ARROW_RIGHT,
            'down': Keys.ARROW_DOWN,
            'rotate': Keys.ARROW_UP,
            'drop': Keys.SPACE,
            'pause': 'p',
            'restart': 'r'  # R키 재시작 추가
        }
        
        if action in action_map:
            if action in ['pause', 'restart']:
                body.send_keys(action_map[action])
            else:
                body.send_keys(action_map[action])
            time.sleep(0.1)  # 액션 간 딜레이
    
    def simple_ai_strategy(self, game_state):
        """간단한 AI 전략"""
        if game_state is None or game_state['game_over']:
            return None
            
        board = game_state['board']
        
        # 간단한 휴리스틱 전략
        # 1. 구멍 최소화
        # 2. 높이 최소화
        # 3. 라인 완성 우선
        
        # 랜덤 액션 (데모용)
        import random
        actions = ['left', 'right', 'down', 'rotate']
        return random.choice(actions)
    
    def play(self, duration=60):
        """AI가 게임을 플레이"""
        print("AI가 테트리스를 플레이합니다...")
        start_time = time.time()
        
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
                
            # AI 전략에 따른 액션 결정
            action = self.simple_ai_strategy(game_state)
            
            if action:
                self.send_action(action)
                print(f"액션: {action}, 점수: {game_state['score']}, 레벨: {game_state['level']}")
            
            time.sleep(0.3)  # 게임 속도 조절
    
    def restart_game(self):
        """게임 재시작"""
        try:
            restart_button = self.driver.find_element(By.CLASS_NAME, "restart-btn")
            restart_button.click()
            time.sleep(1)
        except:
            # 게임오버가 아닌 경우, 페이지 새로고침
            self.driver.refresh()
            time.sleep(3)
    
    def close(self):
        """브라우저 종료"""
        if self.driver:
            self.driver.quit()

def main():
    # 테트리스 AI 초기화
    ai = TetrisAI()
    
    try:
        # 게임 시작
        ai.start_game()
        
        # AI 플레이 (60초간)
        ai.play(duration=60)
        
        # 게임 재시작 후 다시 플레이
        print("\n게임을 재시작합니다...")
        ai.restart_game()
        ai.play(duration=60)
        
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        ai.close()

if __name__ == "__main__":
    main() 