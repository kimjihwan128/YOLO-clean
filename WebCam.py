import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best.pt")  # 학습된 YOLO 모델

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 기본 웹캠 사용

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

def detect_color(cropped_image):
    """신호등의 색상 (빨강, 노랑, 초록)을 감지"""
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # 색상 범위 정의 (HSV)
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    # 색상 범위에 맞는 마스크 생성
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 각 색상의 픽셀 개수 확인
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    # 가장 많은 픽셀을 가진 색상 반환
    if red_count > yellow_count and red_count > green_count:
        return "Red", (0, 0, 255)  # 빨간색
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow", (0, 255, 255)  # 노란색
    elif green_count > red_count and green_count > yellow_count:
        return "Green", (0, 255, 0)  # 초록색
    else:
        return "Unknown", (255, 255, 255)  # 알 수 없음

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델로 추론
    results = model(frame)  # 프레임에 대해 추론 수행

    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 (x1, y1), (x2, y2)
            cropped_image = frame[y1:y2, x1:x2]  # 바운딩 박스 내 이미지 자르기

            # 색상 감지
            traffic_light_color, color_code = detect_color(cropped_image)

            # 결과 출력
            label = f"{traffic_light_color}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_code, 2)  # 바운딩 박스 색상 변경
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_code, 2)

    # 결과 시각화
    cv2.imshow("Traffic Light Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()