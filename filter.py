import json
import os
from shutil import copyfile

# COCO annotations 파일 경로
coco_annotation_path = r'C:\Users\user\OneDrive\Desktop\머신러닝하는조\annotations\instances_train2017.json'

# 신호등 클래스 ID (YOLO에서는 0으로 설정)
TRAFFIC_LIGHT_CLASS_ID = 0

# 출력 디렉토리 설정
output_dir = 'filtered_dataset'
os.makedirs(f'{output_dir}/images', exist_ok=True)
os.makedirs(f'{output_dir}/labels', exist_ok=True)

# COCO 데이터 필터링
with open(coco_annotation_path, 'r') as f:
    coco_data = json.load(f)

filtered_images = set()
filtered_annotations = []

# 신호등 객체만 필터링
for annotation in coco_data['annotations']:
    if annotation['category_id'] == 10:  # COCO 신호등 클래스 ID
        # YOLO 클래스 ID로 변환
        annotation['category_id'] = TRAFFIC_LIGHT_CLASS_ID
        filtered_annotations.append(annotation)
        filtered_images.add(annotation['image_id'])

# 해당 이미지 정보 추출
filtered_image_info = [
    img for img in coco_data['images'] if img['id'] in filtered_images
]

# 이미지 및 라벨 저장
for image_info in filtered_image_info:
    # 이미지 경로 설정
    image_path = f"C:/Users/user/OneDrive/Desktop/머신러닝하는조/train2017/{image_info['file_name']}"
    output_image_path = f"{output_dir}/images/{image_info['file_name']}"

    if os.path.exists(image_path):
        copyfile(image_path, output_image_path)

        # 라벨 파일 생성 (YOLO 형식으로 변환)
        label_path = f"{output_dir}/labels/{image_info['file_name'].replace('.jpg', '.txt')}"
        with open(label_path, 'w') as label_file:
            for annotation in filtered_annotations:
                if annotation['image_id'] == image_info['id']:
                    # 바운딩 박스 좌표 변환 (YOLO 형식)
                    bbox = annotation['bbox']
                    x_center = (bbox[0] + bbox[2] / 2) / image_info['width']
                    y_center = (bbox[1] + bbox[3] / 2) / image_info['height']
                    width = bbox[2] / image_info['width']
                    height = bbox[3] / image_info['height']

                    # YOLO 클래스 ID와 변환된 좌표 저장
                    label_file.write(f"{annotation['category_id']} {x_center} {y_center} {width} {height}\n")
