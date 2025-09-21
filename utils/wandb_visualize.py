"""
W&B 시각화 유틸리티
- 검증/테스트 이미지와 바운딩 박스 시각화
- 실제 vs 예측 결과 비교
- 학습 진행상황 시각적 모니터링
"""

import wandb
import cv2
import numpy as np
from pathlib import Path
import random
from ultralytics import YOLO
import yaml


class WandBVisualizer:
    def __init__(self, class_names):
        """
        W&B 시각화 클래스 초기화

        Args:
            class_names: 클래스 이름 리스트 ['crack', 'efflorescence', 'spalling']
        """
        self.class_names = class_names
        self.colors = self._generate_colors()

    def _generate_colors(self):
        """클래스별 고유 색상 생성"""
        colors = {}
        for i, name in enumerate(self.class_names):
            # 균열=빨강, 백태=파랑, 박리=초록 등 구분되는 색상
            if 'crack' in name.lower():
                colors[name] = (255, 0, 0)  # 빨강
            elif 'efflorescence' in name.lower():
                colors[name] = (0, 0, 255)  # 파랑
            elif 'spalling' in name.lower():
                colors[name] = (0, 255, 0)  # 초록
            else:
                # 기타 클래스는 무작위 색상
                colors[name] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors

    def draw_boxes(self, image, boxes, labels, scores=None, title_prefix=""):
        """
        이미지에 바운딩 박스 그리기

        Args:
            image: numpy 이미지 배열
            boxes: 바운딩 박스 좌표 [[x1,y1,x2,y2], ...]
            labels: 클래스 라벨 [0, 1, 2, ...]
            scores: 신뢰도 점수 [0.95, 0.87, ...]
            title_prefix: 제목 접두사 ("GT" 또는 "Pred")

        Returns:
            annotated_image: 바운딩 박스가 그려진 이미지
        """
        img = image.copy()

        for i, (box, label_idx) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[int(label_idx)]
            color = self.colors[class_name]

            # 바운딩 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 라벨 텍스트
            if scores is not None:
                text = f"{class_name} {scores[i]:.2f}"
            else:
                text = class_name

            # 텍스트 배경
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1-25), (x1+text_size[0], y1), color, -1)

            # 텍스트
            cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return img

    def log_validation_images(self, model, val_dataset_path, num_images=10, epoch=None):
        """
        검증 이미지 예측 결과를 W&B에 로깅

        Args:
            model: 학습된 YOLO 모델
            val_dataset_path: 검증 데이터셋 경로
            num_images: 로깅할 이미지 수
            epoch: 현재 에폭 (선택적)
        """
        val_images_path = Path(val_dataset_path) / 'val' / 'images'
        val_labels_path = Path(val_dataset_path) / 'val' / 'labels'

        if not val_images_path.exists():
            print("⚠️  검증 이미지 폴더를 찾을 수 없습니다.")
            return

        # 랜덤 이미지 선택
        image_files = list(val_images_path.glob('*.[jp][pn][g]'))
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        wandb_images = []

        for img_path in selected_images:
            # 이미지 로드
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Ground Truth 라벨 로드
            label_path = val_labels_path / (img_path.stem + '.txt')
            gt_boxes, gt_labels = self._load_yolo_labels(label_path, w, h)

            # 모델 예측
            results = model.predict(str(img_path), verbose=False)
            pred_boxes = []
            pred_labels = []
            pred_scores = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()
                pred_labels = boxes.cls.cpu().numpy()
                pred_scores = boxes.conf.cpu().numpy()

            # Ground Truth 이미지
            gt_img = self.draw_boxes(image, gt_boxes, gt_labels, title_prefix="GT")

            # 예측 이미지
            pred_img = self.draw_boxes(image, pred_boxes, pred_labels, pred_scores, title_prefix="Pred")

            # 나란히 비교 이미지 생성
            comparison_img = np.hstack([gt_img, pred_img])

            # 제목 추가
            title_img = self._add_title(comparison_img, f"GT vs Pred: {img_path.name}")

            # W&B 이미지 객체 생성
            wandb_img = wandb.Image(
                title_img,
                caption=f"Left: Ground Truth, Right: Prediction ({len(gt_labels)} GT, {len(pred_labels)} Pred)"
            )
            wandb_images.append(wandb_img)

        # W&B에 로깅
        log_key = f"validation_images"
        if epoch is not None:
            log_key += f"_epoch_{epoch}"

        wandb.log({log_key: wandb_images})
        print(f"✅ 검증 이미지 {len(wandb_images)}개를 W&B에 로깅했습니다.")

    def log_test_results(self, model, test_dataset_path, num_images=20):
        """
        테스트 결과를 W&B에 로깅

        Args:
            model: 학습된 YOLO 모델
            test_dataset_path: 테스트 데이터셋 경로
            num_images: 로깅할 이미지 수
        """
        test_images_path = Path(test_dataset_path) / 'test' / 'images'
        test_labels_path = Path(test_dataset_path) / 'test' / 'labels'

        if not test_images_path.exists():
            print("⚠️  테스트 이미지 폴더를 찾을 수 없습니다.")
            return

        # 성능별로 이미지 분류
        good_predictions = []  # 정확한 예측
        bad_predictions = []   # 부정확한 예측

        image_files = list(test_images_path.glob('*.[jp][pn][g]'))
        selected_images = random.sample(image_files, min(num_images*2, len(image_files)))

        for img_path in selected_images:
            # 이미지와 라벨 로드
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            label_path = test_labels_path / (img_path.stem + '.txt')
            gt_boxes, gt_labels = self._load_yolo_labels(label_path, w, h)

            # 예측
            results = model.predict(str(img_path), verbose=False)
            pred_boxes = []
            pred_labels = []
            pred_scores = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()
                pred_labels = boxes.cls.cpu().numpy()
                pred_scores = boxes.conf.cpu().numpy()

            # 예측 품질 평가 (간단한 IoU 기반)
            is_good_prediction = self._evaluate_prediction_quality(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
            )

            # 비교 이미지 생성
            gt_img = self.draw_boxes(image, gt_boxes, gt_labels, title_prefix="GT")
            pred_img = self.draw_boxes(image, pred_boxes, pred_labels, pred_scores, title_prefix="Pred")
            comparison_img = np.hstack([gt_img, pred_img])
            title_img = self._add_title(comparison_img, f"{img_path.name}")

            wandb_img = wandb.Image(
                title_img,
                caption=f"GT: {len(gt_labels)}, Pred: {len(pred_labels)} objects"
            )

            if is_good_prediction and len(good_predictions) < num_images // 2:
                good_predictions.append(wandb_img)
            elif not is_good_prediction and len(bad_predictions) < num_images // 2:
                bad_predictions.append(wandb_img)

            if len(good_predictions) >= num_images // 2 and len(bad_predictions) >= num_images // 2:
                break

        # W&B에 로깅
        if good_predictions:
            wandb.log({"test_good_predictions": good_predictions})
            print(f"✅ 좋은 예측 결과 {len(good_predictions)}개를 로깅했습니다.")

        if bad_predictions:
            wandb.log({"test_challenging_cases": bad_predictions})
            print(f"⚠️  어려운 예측 케이스 {len(bad_predictions)}개를 로깅했습니다.")

    def _load_yolo_labels(self, label_path, img_w, img_h):
        """YOLO 라벨 파일 로드"""
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # YOLO 형식을 절대 좌표로 변환
                    x1 = int((x_center - width/2) * img_w)
                    y1 = int((y_center - height/2) * img_h)
                    x2 = int((x_center + width/2) * img_w)
                    y2 = int((y_center + height/2) * img_h)

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        return boxes, labels

    def _evaluate_prediction_quality(self, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
        """예측 품질 평가 (간단한 IoU 기반)"""
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return True  # 둘 다 없으면 정확
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return False  # 하나만 있으면 부정확

        # 간단한 매칭 (실제로는 Hungarian algorithm 사용 권장)
        matched = 0
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                if abs(gt_label - pred_label) < 0.5 and pred_score > 0.5:  # 같은 클래스
                    iou = self._calculate_iou(gt_box, pred_box)
                    if iou > iou_threshold:
                        matched += 1
                        break

        # 매칭된 비율이 50% 이상이면 좋은 예측
        return matched / len(gt_boxes) >= 0.5

    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        # 교집합 영역
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _add_title(self, image, title):
        """이미지 상단에 제목 추가"""
        title_height = 30
        title_img = np.ones((title_height, image.shape[1], 3), dtype=np.uint8) * 50

        # 제목 텍스트
        cv2.putText(title_img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 제목과 이미지 결합
        return np.vstack([title_img, image])


def log_training_samples(dataset_path, class_names, num_samples=10):
    """
    학습 시작 전 데이터셋 샘플 로깅

    Args:
        dataset_path: 데이터셋 경로
        class_names: 클래스 이름 리스트
        num_samples: 로깅할 샘플 수
    """
    visualizer = WandBVisualizer(class_names)

    train_images_path = Path(dataset_path) / 'train' / 'images'
    train_labels_path = Path(dataset_path) / 'train' / 'labels'

    if not train_images_path.exists():
        print("⚠️  학습 이미지 폴더를 찾을 수 없습니다.")
        return

    image_files = list(train_images_path.glob('*.[jp][pn][g]'))
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))

    wandb_images = []
    class_counts = {name: 0 for name in class_names}

    for img_path in selected_images:
        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 라벨 로드
        label_path = train_labels_path / (img_path.stem + '.txt')
        boxes, labels = visualizer._load_yolo_labels(label_path, w, h)

        # 클래스 카운트
        for label in labels:
            class_counts[class_names[label]] += 1

        # 박스 그리기
        annotated_img = visualizer.draw_boxes(image, boxes, labels, title_prefix="Train")
        title_img = visualizer._add_title(annotated_img, f"Training Sample: {img_path.name}")

        wandb_img = wandb.Image(
            title_img,
            caption=f"Objects: {len(labels)}"
        )
        wandb_images.append(wandb_img)

    # W&B에 로깅
    wandb.log({
        "training_samples": wandb_images,
        "class_distribution": class_counts
    })

    print(f"✅ 학습 샘플 {len(wandb_images)}개를 W&B에 로깅했습니다.")
    print(f"📊 클래스 분포: {class_counts}")