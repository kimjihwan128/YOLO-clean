from ultralytics import YOLO

# YOLO 모델 생성
model = YOLO('yolov8n.pt')  # 사전 학습된 YOLOv8 모델

# 모델 학습
model.train(
    data='data.yaml',       # 데이터 설정 파일
    epochs=50,              # 학습 반복 횟수
    imgsz=640,              # 입력 이미지 크기
    batch=16                # 배치 크기
)
