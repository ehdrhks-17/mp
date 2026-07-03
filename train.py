import os
from ultralytics import YOLO

def create_yaml():
    # 현재 디렉토리 절대 경로
    current_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    
    yaml_content = f"""
path: {dataset_dir}
train: images/train
val: images/train # 검증(Validation)도 동일한 데이터 사용 (단순화)

# 클래스 정보
names:
  0: monster
"""
    yaml_path = os.path.join(current_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path

def train_model(output_name="best"):
    print("🚀 학습 준비 중...")
    yaml_path = create_yaml()
    
    # 1. 가장 가볍고 빠른 YOLOv8n(nano) 기본 모델 로드
    model = YOLO("yolov8n.pt") 
    
    print("🧠 인공지능이 몬스터의 모습을 공부하기 시작합니다! (컴퓨터 사양에 따라 수십 분 소요될 수 있습니다)")
    
    # 2. 모델 학습 시작
    # epochs=50: 사진들을 50번 반복해서 봅니다.
    # imgsz=640: 이미지 크기
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        device="cpu", # 그래픽카드가 없어도 돌아가도록 강제 CPU 설정 (GPU가 있다면 '0'으로 변경 가능)
        plots=False   # 그래프 생성 생략 (속도 향상)
    )
    
    print("✅ 학습이 완료되었습니다!")
    
    # 3. 완성된 가중치 파일(best.pt)을 폴더 밖으로 복사
    try:
        import shutil
        # ultralytics는 결과물을 runs/detect/train/weights/ 에 저장합니다.
        # 학습을 여러 번 돌리면 train2, train3 등 숫자가 올라갑니다.
        # 가장 최근 폴더를 찾습니다.
        runs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "runs", "detect")
        train_folders = [f for f in os.listdir(runs_dir) if f.startswith('train')]
        latest_train = max(train_folders, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
        
        best_pt_path = os.path.join(runs_dir, latest_train, "weights", "best.pt")
        dest_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mobs")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{output_name}.pt")
        
        shutil.copy2(best_pt_path, dest_path)
        print(f"🎉 성공! [{output_name}.pt] 파일이 완성되어 mobs 폴더에 저장되었습니다.")
        print(f"이제 기존의 [main.py]를 실행하시면 인공지능 사냥 봇이 작동합니다!")
        
    except Exception as e:
        print(f"파일 복사 중 오류가 발생했습니다. runs/detect/train/weights/best.pt 파일을 직접 꺼내주세요. 오류: {e}")

if __name__ == "__main__":
    import sys
    out_name = "best"
    if len(sys.argv) > 1:
        out_name = sys.argv[1]
    train_model(out_name)
