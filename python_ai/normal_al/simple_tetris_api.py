import requests
import time
import random

class SimpleTetrisAPI:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/game"
    
    def send_action(self, action):
        """액션을 전송합니다."""
        try:
            response = requests.post(self.api_url, json={
                'type': 'action',
                'action': action
            })
            return response.status_code == 200
        except:
            return False
    
    def get_game_state(self):
        """게임 상태를 가져옵니다."""
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {})
        except:
            pass
        return {}
    
    def check_connection(self):
        """서버 연결을 확인합니다."""
        try:
            response = requests.get(self.api_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def simple_play(self, duration=60):
        """간단한 랜덤 플레이"""
        print("🔍 서버 연결 확인 중...")
        
        if not self.check_connection():
            print("❌ 서버에 연결할 수 없습니다.")
            print("💡 다음을 확인하세요:")
            print("   1. 'npm run dev'로 Next.js 서버가 실행 중인지")
            print("   2. http://localhost:3000에서 게임이 로드되는지")
            print("   3. 브라우저에서 'AI 모드 ON' 버튼을 클릭했는지")
            return
        
        print("✅ 서버 연결 성공!")
        print(f"🤖 {duration}초간 랜덤 플레이를 시작합니다...")
        print("🎮 브라우저에서 게임을 확인하세요!")
        
        actions = ['left', 'right', 'down', 'rotate', 'drop']
        start_time = time.time()
        action_count = 0
        
        while time.time() - start_time < duration:
            # 랜덤 액션 선택
            action = random.choice(actions)
            
            # 액션 전송
            if self.send_action(action):
                action_count += 1
                print(f"🎮 액션 #{action_count}: {action}")
            
            # 게임 상태 확인
            game_state = self.get_game_state()
            if game_state.get('isGameOver'):
                print(f"🎯 게임 오버! 점수: {game_state.get('score', 0)}")
                break
            
            time.sleep(0.5)  # 0.5초마다 액션
        
        print(f"✅ 플레이 완료! 총 {action_count}개 액션 전송")

def main():
    print("🎮 간단한 테트리스 API 클라이언트")
    print("=" * 40)
    
    api = SimpleTetrisAPI()
    
    try:
        api.simple_play(duration=120)  # 2분간 플레이
    except KeyboardInterrupt:
        print("\n⏹️  중단되었습니다.")

if __name__ == "__main__":
    main() 