#!/usr/bin/env python3
"""
테트리스 API 연결 테스트 스크립트
"""
import requests
import json

def test_connection():
    """API 서버 연결을 테스트합니다."""
    api_url = "http://localhost:3000/api/game"
    
    print("🔍 테트리스 API 연결 테스트")
    print("=" * 40)
    
    try:
        # GET 요청 테스트
        print("1. GET 요청 테스트...")
        response = requests.get(api_url, timeout=5)
        print(f"   상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   응답: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print("   ✅ GET 요청 성공!")
        else:
            print(f"   ❌ GET 요청 실패: {response.status_code}")
            return False
        
        # POST 요청 테스트
        print("\n2. POST 요청 테스트...")
        test_payload = {
            "type": "action",
            "action": "down"
        }
        
        response = requests.post(api_url, json=test_payload, timeout=5)
        print(f"   상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   응답: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print("   ✅ POST 요청 성공!")
        else:
            print(f"   ❌ POST 요청 실패: {response.status_code}")
            return False
        
        print("\n🎉 모든 테스트 통과!")
        print("\n💡 다음 단계:")
        print("   1. 브라우저에서 http://localhost:3000 접속")
        print("   2. '🤖 AI 모드 ON' 버튼 클릭")
        print("   3. Python AI 스크립트 실행")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: 서버에 연결할 수 없습니다.")
        print("\n💡 해결책:")
        print("   1. Next.js 서버가 실행 중인지 확인: 'pnpm dev'")
        print("   2. http://localhost:3000에서 게임이 로드되는지 확인")
        print("   3. 포트 3000이 사용 중인지 확인")
        
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: 서버 응답이 너무 느립니다.")
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
    
    return False

def main():
    success = test_connection()
    
    if success:
        # 간단한 액션 테스트
        print("\n🎮 간단한 액션 테스트 (5개 액션 전송)")
        api_url = "http://localhost:3000/api/game"
        actions = ['left', 'right', 'down', 'rotate', 'drop']
        
        for i, action in enumerate(actions, 1):
            try:
                response = requests.post(api_url, json={
                    'type': 'action',
                    'action': action
                })
                
                if response.status_code == 200:
                    print(f"   {i}. {action} ✅")
                else:
                    print(f"   {i}. {action} ❌")
                    
            except Exception as e:
                print(f"   {i}. {action} ❌ ({e})")
        
        print("\n✅ 테스트 완료! AI 스크립트를 실행해보세요.")
    else:
        print("\n❌ 연결 테스트 실패. Next.js 서버를 먼저 실행하세요.")

if __name__ == "__main__":
    main() 