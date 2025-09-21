#!/usr/bin/env python3
"""
완벽한 YOLO 변환 시스템
- JSON bbox [x,y,w,h] → YOLO [center_x, center_y, width, height] (정규화)
- 3가지 클래스만 정확히 매핑: ConcreteCrack(0), Efflorescene(1), Spalling(2)
- 좌표 범위 검증 및 클리핑
"""

import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional

class PerfectYOLOConverter:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        # 3가지 클래스만 정확히 매핑
        self.class_mapping = {
            'ConcreteCrack': 0,      # 균열
            'Efflorescene': 1,       # 백태누수 (오타 있음)
            'Efflorescence': 1,      # 백태누수 (올바른 철자)
            'Spalling': 2            # 박리
        }

        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {0: 0, 1: 0, 2: 0},
            'invalid_boxes': 0,
            'skipped_classes': 0,
            'coordinate_errors': []
        }

    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int, filename: str = "") -> Optional[Tuple[float, float, float, float]]:
        """
        JSON bbox [x, y, width, height] → YOLO [center_x, center_y, width, height] (정규화)
        간단하고 정확한 변환 - 과도한 클리핑 제거
        """
        try:
            x, y, w, h = bbox

            # 기본 유효성 검사만
            if w <= 0 or h <= 0:
                self.stats['coordinate_errors'].append(f"{filename}: 잘못된 크기 bbox [{x}, {y}, {w}, {h}]")
                return None

            # 이미지 경계를 벗어나는 bbox는 건너뛰기 (클리핑하지 않음)
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                self.stats['coordinate_errors'].append(f"{filename}: 경계 초과 bbox 건너뛰기 [{x}, {y}, {w}, {h}]")
                return None

            # 간단한 YOLO 변환
            # 1. 중심점 계산
            center_x = x + w / 2
            center_y = y + h / 2

            # 2. 정규화 (0~1 범위)
            norm_center_x = center_x / img_width
            norm_center_y = center_y / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            # 3. 최종 범위 확인 (0~1 범위 보장)
            if not (0 <= norm_center_x <= 1 and 0 <= norm_center_y <= 1 and
                    0 < norm_width <= 1 and 0 < norm_height <= 1):
                self.stats['coordinate_errors'].append(f"{filename}: 정규화 범위 오류")
                return None

            return norm_center_x, norm_center_y, norm_width, norm_height

        except Exception as e:
            self.stats['coordinate_errors'].append(f"{filename}: 변환 에러 {str(e)}")
            return None

    def get_class_id(self, class_name: str) -> Optional[int]:
        """클래스 이름을 YOLO ID로 변환"""
        return self.class_mapping.get(class_name, None)

    def process_json_file(self, json_path: Path) -> List[str]:
        """단일 JSON 파일을 처리하여 YOLO 형식 라인들 반환"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON 읽기 실패 {json_path}: {e}")
            return []

        if 'images' not in data or 'annotations' not in data:
            print(f"잘못된 JSON 구조: {json_path}")
            return []

        # 이미지 정보 추출
        image_info = data['images'][0]  # 첫 번째 이미지 정보 사용
        img_width = image_info['width']
        img_height = image_info['height']

        yolo_lines = []
        filename = json_path.stem

        for annotation in data['annotations']:
            # 클래스 정보 추출
            if 'attributes' not in annotation or 'class' not in annotation['attributes']:
                self.stats['skipped_classes'] += 1
                continue

            class_name = annotation['attributes']['class']
            class_id = self.get_class_id(class_name)

            if class_id is None:
                print(f"알 수 없는 클래스 '{class_name}' 건너뜀: {json_path}")
                self.stats['skipped_classes'] += 1
                continue

            # 바운딩 박스 변환
            bbox = annotation['bbox']
            yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height, filename)

            if yolo_bbox is None:
                self.stats['invalid_boxes'] += 1
                continue

            # YOLO 형식으로 저장
            center_x, center_y, width, height = yolo_bbox
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)

            self.stats['class_counts'][class_id] += 1
            self.stats['total_annotations'] += 1

        return yolo_lines

    def convert_dataset(self, output_path: str):
        """전체 데이터셋 변환 및 분할"""
        output_path = Path(output_path)

        # 출력 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Train 폴더에서 모든 데이터 수집
        train_path = self.dataset_path / 'Train'
        labels_path = train_path / 'labels'
        images_path = train_path / 'images'

        if not labels_path.exists() or not images_path.exists():
            print(f"필요한 폴더가 없습니다: {labels_path}, {images_path}")
            return

        print(f"\n=== Train 폴더에서 데이터 수집 중 ===")
        print(f"Labels: {labels_path}")
        print(f"Images: {images_path}")

        # JSON 파일들 처리
        json_files = list(labels_path.glob('*.json'))
        print(f"JSON 파일 수: {len(json_files)}")

        if not json_files:
            print("JSON 파일을 찾을 수 없습니다!")
            return

        # 이미지-라벨 페어로 데이터 수집
        image_label_pairs = []

        for json_file in json_files:
            # JSON에서 실제 이미지 파일명 가져오기
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                if 'images' not in json_data or not json_data['images']:
                    print(f"이미지 정보 없음: {json_file}")
                    continue

                # JSON 내부의 실제 이미지 파일명
                image_filename = json_data['images'][0]['file_name']
                image_base_name = Path(image_filename).stem

            except Exception as e:
                print(f"JSON 읽기 실패 {json_file}: {e}")
                continue

            # JSON에서 찾은 이미지 파일명으로 실제 이미지 파일 찾기
            image_file = None
            for ext in ['.tiff', '.jpg', '.jpeg', '.png', '.TIFF', '.JPG', '.JPEG', '.PNG']:
                potential_image = images_path / f"{image_base_name}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break

            if image_file is None:
                print(f"대응 이미지 없음: {image_filename} -> {image_base_name}")
                continue

            # JSON → YOLO 변환
            yolo_lines = self.process_json_file(json_file)

            if not yolo_lines:
                print(f"변환 실패 또는 빈 파일: {json_file}")
                continue

            # 이미지-라벨 페어 저장 (같은 base_name 사용)
            image_label_pairs.append({
                'base_name': image_base_name,  # JSON에서 가져온 이미지 base_name 사용
                'yolo_lines': yolo_lines,
                'image_file': image_file
            })

        print(f"수집된 이미지-라벨 페어: {len(image_label_pairs)}개")

        # 페어 단위로 데이터 분할 (80:15:5)
        import random
        random.seed(42)  # 재현 가능한 분할
        random.shuffle(image_label_pairs)

        total_pairs = len(image_label_pairs)
        train_count = int(total_pairs * 0.8)
        val_count = int(total_pairs * 0.15)
        test_count = total_pairs - train_count - val_count

        train_pairs = image_label_pairs[:train_count]
        val_pairs = image_label_pairs[train_count:train_count + val_count]
        test_pairs = image_label_pairs[train_count + val_count:]

        print(f"페어 분할 결과: Train {len(train_pairs)}, Val {len(val_pairs)}, Test {len(test_pairs)}")

        # 각 분할별로 저장 (페어 단위로 유지)
        for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            print(f"\n=== {split_name.upper()} 페어 저장 중 ===")

            for i, pair in enumerate(split_pairs):
                base_name = pair['base_name']
                yolo_lines = pair['yolo_lines']
                image_file = pair['image_file']

                # YOLO 텍스트 파일 저장 (같은 base_name으로)
                output_txt = output_path / split_name / 'labels' / f"{base_name}.txt"
                with open(output_txt, 'w', encoding='utf-8') as f:
                    for line in yolo_lines:
                        f.write(line + '\n')

                # 이미지 파일 복사 (같은 base_name으로)
                output_img = output_path / split_name / 'images' / f"{base_name}.jpg"

                if image_file.suffix.lower() in ['.tiff', '.tif']:
                    # TIFF → JPG 변환
                    try:
                        from PIL import Image
                        with Image.open(image_file) as img:
                            rgb_img = img.convert('RGB')
                            rgb_img.save(output_img, 'JPEG', quality=95)
                    except Exception as e:
                        print(f"이미지 변환 실패 {image_file}: {e}")
                        continue
                else:
                    # 일반 복사
                    shutil.copy2(image_file, output_img)

                self.stats['total_images'] += 1

                if (i + 1) % 100 == 0:
                    print(f"{split_name} 저장: {i + 1}/{len(split_pairs)}")

            print(f"{split_name.upper()} 완료: {len(split_pairs)}개 페어 저장됨")

    def print_statistics(self):
        """변환 통계 출력"""
        print("\n" + "="*50)
        print("변환 완료 통계")
        print("="*50)
        print(f"총 이미지 수: {self.stats['total_images']}")
        print(f"총 어노테이션 수: {self.stats['total_annotations']}")
        print(f"잘못된 박스 수: {self.stats['invalid_boxes']}")
        print(f"건너뛴 클래스 수: {self.stats['skipped_classes']}")
        print("\n클래스별 분포:")
        print(f"  0 (ConcreteCrack): {self.stats['class_counts'][0]}")
        print(f"  1 (Efflorescene): {self.stats['class_counts'][1]}")
        print(f"  2 (Spalling): {self.stats['class_counts'][2]}")

        if self.stats['coordinate_errors']:
            print(f"\n좌표 에러 {len(self.stats['coordinate_errors'])}개:")
            for error in self.stats['coordinate_errors'][:10]:  # 처음 10개만 표시
                print(f"  {error}")
            if len(self.stats['coordinate_errors']) > 10:
                print(f"  ... 추가 {len(self.stats['coordinate_errors']) - 10}개 에러")

def main():
    # 데이터셋 경로 설정
    dataset_path = "/Users/hyunwoo/Desktop/DroneProject/yolov11_project/rawdata"
    output_path = "/Users/hyunwoo/Desktop/DroneProject/yolov11_project/dataset"

    print("완벽한 YOLO 변환 시스템 시작")
    print("3가지 클래스만 변환: ConcreteCrack(0), Efflorescene(1), Spalling(2)")
    print(f"입력: {dataset_path}")
    print(f"출력: {output_path}")

    converter = PerfectYOLOConverter(dataset_path)
    converter.convert_dataset(output_path)
    converter.print_statistics()

    print("\n변환 완료!")

if __name__ == "__main__":
    main()