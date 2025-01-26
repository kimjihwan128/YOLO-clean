from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('C:/Users/user/OneDrive/Desktop/머신러닝하는조/runs/detect/train/weights/best.pt')

# 이미지 테스트
results = model.predict(source='test_image.jpg', save=True)
print(results)
