"""
데이터 처리 유틸리티 (새 버전)
- perfect_yolo_converter.py와 split_dataset.py 기능 통합
- train.py에서 필요한 함수들 제공
"""

import json
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_dataset(images_dir, labels_dir, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05, seed=42):
    """
    데이터셋을 train/val/test로 분할 (새 버전 - TXT 라벨 기준)

    Args:
        images_dir: 이미지 디렉토리 경로
        labels_dir: 라벨 디렉토리 경로 (TXT 파일들)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드

    Returns:
        train_stems, val_stems, test_stems: 각 세트의 파일명 리스트
    """
    random.seed(seed)

    # 이미지 파일 목록
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []

    print(f"이미지 디렉토리: {images_dir}")
    for ext in image_extensions:
        found_files = list(Path(images_dir).glob(f'*{ext}'))
        if found_files:
            print(f"  {ext}: {len(found_files)}개")
        images.extend(found_files)

    print(f"총 이미지: {len(images)}개")

    # 라벨 파일 확인 (TXT 형식)
    valid_pairs = []
    for img in images:
        txt_label = Path(labels_dir) / f"{img.stem}.txt"
        if txt_label.exists():
            valid_pairs.append(img.stem)

    print(f"유효한 이미지-라벨 쌍: {len(valid_pairs)}개")

    if len(valid_pairs) == 0:
        print("⚠️ 경고: 유효한 데이터 쌍이 없습니다!")
        print("  - 이미지와 라벨 파일명이 일치하는지 확인하세요")
        print("  - 라벨이 .txt 형식인지 확인하세요")
        return [], [], []

    # 클래스별 분포 확인 (균등 분할을 위해)
    class_files = defaultdict(list)

    for stem in valid_pairs:
        label_file = Path(labels_dir) / f"{stem}.txt"
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if lines:
            # 첫 번째 클래스로 분류 (대표 클래스)
            first_class = int(lines[0].split()[0]) if lines[0].strip() else 0
            class_files[first_class].append(stem)

    print("\n클래스별 분포:")
    class_names = {0: "ConcreteCrack", 1: "Efflorescence", 2: "Spalling"}
    for class_id, files in class_files.items():
        print(f"  {class_id} ({class_names.get(class_id, 'Unknown')}): {len(files)}개")

    # 각 클래스별로 균등 분할
    train_stems = []
    val_stems = []
    test_stems = []

    for class_id, files in class_files.items():
        random.shuffle(files)

        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_stems.extend(files[:train_end])
        val_stems.extend(files[train_end:val_end])
        test_stems.extend(files[val_end:])

    # 각 세트 내에서 셔플
    random.shuffle(train_stems)
    random.shuffle(val_stems)
    random.shuffle(test_stems)

    print(f"\n데이터 분할 결과:")
    print(f"  Train: {len(train_stems)}개 ({len(train_stems)/len(valid_pairs)*100:.1f}%)")
    print(f"  Val: {len(val_stems)}개 ({len(val_stems)/len(valid_pairs)*100:.1f}%)")
    print(f"  Test: {len(test_stems)}개 ({len(test_stems)/len(valid_pairs)*100:.1f}%)")

    return train_stems, val_stems, test_stems


def prepare_split_folders(base_dir, train_stems, val_stems, test_stems, images_dir, labels_dir, class_names):
    """
    분할된 데이터를 각 폴더로 복사 (TXT 라벨 사용)

    Args:
        base_dir: 기본 디렉토리 (dataset_split 또는 dataset_perfect)
        train_stems, val_stems, test_stems: 각 세트의 파일명 리스트
        images_dir: 원본 이미지 디렉토리
        labels_dir: 원본 라벨 디렉토리 (TXT 파일들)
        class_names: 클래스 이름 리스트
    """
    splits = {
        'train': train_stems,
        'val': val_stems,
        'test': test_stems
    }

    for split_name, stems in splits.items():
        # 디렉토리 생성
        split_images_dir = base_dir / split_name / 'images'
        split_labels_dir = base_dir / split_name / 'labels'
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{split_name} 데이터 준비 중...")
        copied = 0

        for stem in stems:
            # 이미지 파일 찾기 및 복사
            image_copied = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                src_img = Path(images_dir) / f"{stem}{ext}"
                if src_img.exists():
                    dst_img = split_images_dir / f"{stem}{ext}"
                    shutil.copy2(src_img, dst_img)
                    image_copied = True
                    break

            # 라벨 파일 복사 (TXT)
            src_label = Path(labels_dir) / f"{stem}.txt"
            if src_label.exists() and image_copied:
                dst_label = split_labels_dir / f"{stem}.txt"
                shutil.copy2(src_label, dst_label)
                copied += 1

        print(f"  {split_name}: {copied}개 파일 복사 완료")


def create_data_yaml(base_dir, class_names, yaml_path):
    """
    YOLOv11용 데이터 설정 파일 생성 (표준 YOLO 형식)

    Args:
        base_dir: 데이터셋 기본 디렉토리
        class_names: 클래스 이름 리스트
        yaml_path: 저장할 YAML 파일 경로
    """
    # 표준 YOLO 형식 (TXT 라벨 사용)
    data_config = {
        'path': str(base_dir.absolute()),
        'train': 'train',  # train/images, train/labels
        'val': 'val',      # val/images, val/labels
        'test': 'test',    # test/images, test/labels
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ YOLO 데이터 설정 파일 생성: {yaml_path}")
    print(f"  클래스 수: {len(class_names)}")
    print(f"  클래스: {class_names}")


def validate_dataset(dataset_path, class_names):
    """
    데이터셋 유효성 검사 (TXT 라벨 기준)

    Args:
        dataset_path: 데이터셋 경로
        class_names: 클래스 이름 리스트

    Returns:
        validation_report: 검증 보고서
    """
    report = {}

    for split in ['train', 'val', 'test']:
        images_dir = Path(dataset_path) / split / 'images'
        labels_dir = Path(dataset_path) / split / 'labels'

        if not images_dir.exists() or not labels_dir.exists():
            continue

        # 이미지 및 라벨 파일 수
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))

        # 매칭 확인
        matched_pairs = 0
        class_counts = {i: 0 for i in range(len(class_names))}
        total_boxes = 0

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                matched_pairs += 1

                # 클래스 및 박스 카운트
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_id] += 1
                                total_boxes += 1

        # 클래스명으로 변환
        class_distribution = {}
        for class_id, count in class_counts.items():
            class_distribution[class_names[class_id]] = count

        report[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'matched_pairs': matched_pairs,
            'total_boxes': total_boxes,
            'class_distribution': class_distribution
        }

    return report


def print_dataset_summary(report):
    """데이터셋 요약 출력"""
    print("\n" + "=" * 60)
    print("              데이터셋 요약")
    print("=" * 60)

    total_images = 0
    total_boxes = 0

    for split, data in report.items():
        print(f"\n[{split.upper()}]")
        print(f"  이미지: {data['images']}개")
        print(f"  라벨: {data['labels']}개")
        print(f"  매칭: {data['matched_pairs']}개")
        print(f"  바운딩박스: {data.get('total_boxes', 0)}개")

        if data['class_distribution']:
            print("  클래스 분포:")
            for class_name, count in data['class_distribution'].items():
                if count > 0:
                    print(f"    - {class_name}: {count}개")

        total_images += data['images']
        total_boxes += data.get('total_boxes', 0)

    print(f"\n[전체 요약]")
    print(f"  총 이미지: {total_images}개")
    print(f"  총 바운딩박스: {total_boxes}개")
    print("=" * 60)


# JSON 관련 함수들은 제거 (더 이상 사용하지 않음)
def create_unified_coco_json(split_dir, split_name, class_names):
    """더 이상 사용하지 않음 - perfect_yolo_converter.py 사용"""
    print(f"⚠️ create_unified_coco_json은 더 이상 사용하지 않습니다.")
    print(f"   대신 perfect_yolo_converter.py를 사용하세요.")
    return None