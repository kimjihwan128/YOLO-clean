import os
import shutil
from sklearn.model_selection import train_test_split

# 이미지와 라벨 경로
image_dir = 'filtered_dataset/images'
label_dir = 'filtered_dataset/labels'

# 파일 리스트 가져오기
images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
labels = [f.replace('.jpg', '.txt') for f in images]

# Train/Val Split
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Train/Val 디렉토리 생성
os.makedirs('filtered_dataset/images/train', exist_ok=True)
os.makedirs('filtered_dataset/images/val', exist_ok=True)
os.makedirs('filtered_dataset/labels/train', exist_ok=True)
os.makedirs('filtered_dataset/labels/val', exist_ok=True)

# 파일 이동
for img, lbl in zip(train_images, train_labels):
    shutil.move(os.path.join(image_dir, img), 'filtered_dataset/images/train/')
    shutil.move(os.path.join(label_dir, lbl), 'filtered_dataset/labels/train/')
for img, lbl in zip(val_images, val_labels):
    shutil.move(os.path.join(image_dir, img), 'filtered_dataset/images/val/')
    shutil.move(os.path.join(label_dir, lbl), 'filtered_dataset/labels/val/')
