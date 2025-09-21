import os
import shutil
from pathlib import Path

def match_and_move_labels(image_folder, all_labels_folder, output_labels_folder):
    """
    이미지 폴더의 파일명과 매칭되는 라벨 파일을 찾아서 이동하는 함수
    
    Args:
        image_folder: 이미지 파일들이 있는 폴더
        all_labels_folder: 모든 라벨이 있는 원본 폴더
        output_labels_folder: 매칭된 라벨들을 이동할 폴더
    """
    
    # 출력 폴더 생성
    Path(output_labels_folder).mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일명들 수집 (확장자 제거)
    image_files = set()
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    print(f"이미지 폴더 스캔 중: {image_folder}")
    for file_path in Path(image_folder).iterdir():
        if file_path.suffix.lower() in image_extensions:
            # 확장자 제거한 파일명
            base_name = file_path.stem
            image_files.add(base_name)
    
    print(f"발견된 이미지 파일: {len(image_files)}개")
    
    # 라벨 파일들 검사 및 이동
    matched_count = 0
    not_found_count = 0
    label_extensions = {'.json', '.xml', '.txt'}
    
    print(f"라벨 폴더 스캔 중: {all_labels_folder}")
    
    for label_path in Path(all_labels_folder).iterdir():
        if label_path.suffix.lower() in label_extensions:
            # 라벨 파일의 확장자 제거한 이름
            label_base_name = label_path.stem
            
            # 매칭되는 이미지가 있는지 확인
            if label_base_name in image_files:
                # 라벨 파일 이동
                destination = Path(output_labels_folder) / label_path.name
                shutil.copy2(str(label_path), str(destination))  # copy2로 변경 (원본 보존)
                matched_count += 1
                if matched_count <= 5:  # 처음 5개만 출력
                    print(f"매칭됨: {label_path.name}")
                elif matched_count % 1000 == 0:  # 1000개마다 진행상황 출력
                    print(f"진행상황: {matched_count}개 처리됨...")
    
    # 매칭되지 않은 이미지들 확인
    matched_labels = set()
    for label_path in Path(all_labels_folder).iterdir():
        if label_path.suffix.lower() in label_extensions:
            matched_labels.add(label_path.stem)
    
    unmatched_images = image_files - matched_labels
    not_found_count = len(unmatched_images)
    
    print(f"\n=== 매칭 결과 ===")
    print(f"총 이미지 파일: {len(image_files)}개")
    print(f"매칭된 라벨: {matched_count}개")
    print(f"매칭 안된 이미지: {not_found_count}개")
    if len(image_files) > 0:
        print(f"매칭률: {matched_count/len(image_files)*100:.1f}%")
    else:
        print("경고: 이미지 파일이 없습니다. 경로를 확인하세요.")
    
    if not_found_count > 0:
        print(f"\n매칭 안된 이미지 예시 (최대 5개):")
        for img in list(unmatched_images)[:5]:
            print(f"  - {img}")
    
    return matched_count, not_found_count

def batch_process_all_folders(base_image_folder, all_labels_folder, base_output_folder):
    """
    여러 배치 폴더를 한번에 처리하는 함수
    """
    
    base_image_path = Path(base_image_folder)
    total_matched = 0
    total_not_found = 0
    
    # 각 배치 폴더 처리
    for batch_folder in base_image_path.iterdir():
        if batch_folder.is_dir():
            print(f"\n{'='*20} {batch_folder.name} {'='*20}")
            
            # 출력 폴더 경로
            output_labels_folder = Path(base_output_folder) / f"{batch_folder.name}_labels"
            
            # 매칭 및 이동 실행
            matched, not_found = match_and_move_labels(
                str(batch_folder),
                all_labels_folder, 
                str(output_labels_folder)
            )
            
            total_matched += matched
            total_not_found += not_found
    
    print(f"\n{'='*20} 전체 결과 {'='*20}")
    print(f"전체 매칭된 라벨: {total_matched}개")
    print(f"전체 매칭 안된 이미지: {total_not_found}개")

def verify_matching(image_folder, label_folder):
    """
    이미지와 라벨이 제대로 매칭되었는지 검증하는 함수
    """
    
    # 이미지 파일명들
    image_names = set()
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    for file_path in Path(image_folder).iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_names.add(file_path.stem)
    
    # 라벨 파일명들  
    label_names = set()
    label_extensions = {'.json', '.xml', '.txt'}
    for file_path in Path(label_folder).iterdir():
        if file_path.suffix.lower() in label_extensions:
            label_names.add(file_path.stem)
    
    # 매칭 결과
    matched = image_names & label_names
    only_images = image_names - label_names
    only_labels = label_names - image_names
    
    print(f"\n=== 매칭 검증 결과 ===")
    print(f"정상 매칭: {len(matched)}개")
    print(f"라벨 없는 이미지: {len(only_images)}개")
    print(f"이미지 없는 라벨: {len(only_labels)}개")
    print(f"매칭률: {len(matched)/(len(matched)+len(only_images))*100:.1f}%")
    
    if only_images and len(only_images) <= 10:
        print(f"라벨 없는 이미지들: {list(only_images)}")
    if only_labels and len(only_labels) <= 10:
        print(f"이미지 없는 라벨들: {list(only_labels)}")

# 사용 예시
if __name__ == "__main__":
    
    # === 실제 경로로 수정하세요 ===
    
    # 1. 단일 폴더 처리 (첫 번째 배치)
    image_folder = "RawData/원천데이터_JPG/박리/콘크리트_박리_원천_19"  # JPG 변환된 이미지 폴더
    all_labels_folder = "RawData/라벨링데이터/박리/콘크리트_박리_라벨링_02"  # 라벨 파일들이 있는 폴더
    output_labels_folder = "RawData/Label/batch1_labels"
    
    print("이미지-라벨 매칭 시작...")
    matched_count, not_found_count = match_and_move_labels(
        image_folder, 
        all_labels_folder, 
        output_labels_folder
    )
    
    # 매칭 검증
    print("\n매칭 검증 중...")
    verify_matching(image_folder, output_labels_folder)
    
    # === 여러 배치 폴더가 있는 경우 ===
    # base_image_folder = "RawData/원천데이터_JPG/균열"  # batch1, batch2, batch3... 포함
    # all_labels_folder = "RawData/라벨링데이터/균열/콘크리트_콘크리트균열_라벨링"
    # base_output_folder = "RawData/Label"
    # 
    # print("전체 배치 처리 시작...")
    # batch_process_all_folders(base_image_folder, all_labels_folder, base_output_folder)