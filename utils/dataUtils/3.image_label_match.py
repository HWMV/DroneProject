import os
import glob
from pathlib import Path

def check_image_label_matching(dataset_path):
    """이미지와 라벨 파일 매칭 상태를 확인하는 함수"""
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"⚠️ {split}: 폴더 없음 (images: {os.path.exists(images_dir)}, labels: {os.path.exists(labels_dir)})")
            continue
        
        # 이미지 파일들 수집
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
        image_files = set()
        
        for ext in image_extensions:
            for img_path in glob.glob(os.path.join(images_dir, ext)):
                filename = Path(img_path).stem  # 확장자 제거
                image_files.add(filename)
        
        # 라벨 파일들 수집
        label_files = set()
        for txt_path in glob.glob(os.path.join(labels_dir, '*.txt')):
            filename = Path(txt_path).stem
            label_files.add(filename)
        
        # 매칭 분석
        matched = image_files & label_files
        only_images = image_files - label_files
        only_labels = label_files - image_files
        
        print(f"\n📊 {split.upper()} 매칭 결과:")
        print(f"   이미지 파일: {len(image_files):,}개")
        print(f"   라벨 파일: {len(label_files):,}개")
        print(f"   정상 매칭: {len(matched):,}개")
        print(f"   라벨 없는 이미지: {len(only_images):,}개")
        print(f"   이미지 없는 라벨: {len(only_labels):,}개")
        
        if only_images:
            issues.append(f"{split}: {len(only_images)}개 이미지에 라벨 없음")
            print(f"   라벨 없는 이미지 예시: {list(only_images)[:3]}")
        
        if only_labels:
            issues.append(f"{split}: {len(only_labels)}개 라벨에 이미지 없음")
            print(f"   이미지 없는 라벨 예시: {list(only_labels)[:3]}")
        
        # 빈 라벨 파일 확인
        empty_labels = []
        for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
            if os.path.getsize(label_file) == 0:
                empty_labels.append(Path(label_file).name)
        
        if empty_labels:
            issues.append(f"{split}: {len(empty_labels)}개 빈 라벨 파일")
            print(f"   빈 라벨 파일: {empty_labels[:3]}")
    
    return issues

def check_cache_files(dataset_path):
    """캐시 파일을 확인하고 삭제하는 함수"""
    
    cache_files = []
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        cache_file = os.path.join(labels_dir + '.cache')
        
        if os.path.exists(cache_file):
            cache_files.append(cache_file)
            print(f"🗂️ 캐시 파일 발견: {cache_file}")
    
    if cache_files:
        response = input(f"\n캐시 파일 {len(cache_files)}개를 삭제하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            for cache_file in cache_files:
                os.remove(cache_file)
                print(f"🗑️ 삭제됨: {cache_file}")
            print("✅ 모든 캐시 파일이 삭제되었습니다. 다시 학습을 시도해보세요.")

def check_dataset_yaml(yaml_path):
    """YAML 파일의 경로가 실제로 존재하는지 확인"""
    
    if not os.path.exists(yaml_path):
        print(f"❌ YAML 파일이 없습니다: {yaml_path}")
        return False
    
    print(f"✅ YAML 파일 존재: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print("📄 YAML 내용:")
        print(content)
    
    return True

def main():
    dataset_path = "/Users/hyunwoo/Desktop/DroneProject/yolov11_project/dataset"
    yaml_path = "/Users/hyunwoo/Desktop/DroneProject/yolov11_project/configs/data.yaml"
    
    print("🔍 이미지-라벨 매칭 검증 시작...")
    
    # 1. 이미지-라벨 매칭 확인
    issues = check_image_label_matching(dataset_path)
    
    # 2. YAML 파일 확인
    print(f"\n🔍 YAML 파일 검증...")
    check_dataset_yaml(yaml_path)
    
    # 3. 캐시 파일 확인 및 삭제
    print(f"\n🔍 캐시 파일 확인...")
    check_cache_files(dataset_path)
    
    # 결과 요약
    print(f"\n📋 전체 검증 결과:")
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    else:
        print("   ✅ 이미지-라벨 매칭에 문제 없음")
    
    print(f"\n💡 권장사항:")
    print(f"   1. 캐시 파일을 삭제했다면 다시 학습 시도")
    print(f"   2. validation 없이 학습해보기: --val False 옵션 추가")
    print(f"   3. 더 작은 배치 크기로 시도: --batch 4")

if __name__ == "__main__":
    main()