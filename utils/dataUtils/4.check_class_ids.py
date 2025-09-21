import os
import numpy as np
from collections import Counter
import glob

def validate_yolo_labels(dataset_path):
    """YOLO 라벨 파일들의 문제점을 찾고 수정하는 함수"""
    
    issues = []
    all_classes = []
    problematic_files = []
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"❌ {labels_dir} 폴더가 없습니다.")
            continue
            
        print(f"\n🔍 {split} 라벨 검증 중...")
        
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # 빈 줄 건너뛰기
                        continue
                    
                    parts = line.split()
                    
                    # 형식 검사
                    if len(parts) != 5:
                        issues.append(f"{label_file}:{line_num} - {len(parts)}개 컬럼 (5개 필요)")
                        problematic_files.append(label_file)
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        # 클래스 ID 검사
                        if class_id < 0:
                            issues.append(f"{label_file}:{line_num} - 음수 클래스 ID: {class_id}")
                            problematic_files.append(label_file)
                        elif class_id > 2:  # 0, 1, 2만 유효
                            issues.append(f"{label_file}:{line_num} - 유효하지 않은 클래스 ID: {class_id}")
                            problematic_files.append(label_file)
                        else:
                            all_classes.append(class_id)
                        
                        # 좌표 검사
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            issues.append(f"{label_file}:{line_num} - 좌표 범위 초과: {x},{y},{w},{h}")
                            problematic_files.append(label_file)
                        
                        # 박스 크기 검사
                        if w <= 0 or h <= 0:
                            issues.append(f"{label_file}:{line_num} - 잘못된 박스 크기: w={w}, h={h}")
                            problematic_files.append(label_file)
                            
                    except ValueError as e:
                        issues.append(f"{label_file}:{line_num} - 파싱 오류: {str(e)}")
                        problematic_files.append(label_file)
                        
            except Exception as e:
                issues.append(f"{label_file} - 파일 읽기 오류: {str(e)}")
                problematic_files.append(label_file)
    
    # 클래스 분포 출력
    if all_classes:
        class_counts = Counter(all_classes)
        print(f"\n📊 클래스 분포:")
        for class_id in sorted(class_counts.keys()):
            print(f"   클래스 {class_id}: {class_counts[class_id]:,}개")
        print(f"   총 객체 수: {len(all_classes):,}개")
    
    return issues, list(set(problematic_files))

def fix_label_files(problematic_files):
    """문제가 있는 라벨 파일들을 수정하는 함수"""
    
    for file_path in problematic_files:
        print(f"\n🔧 수정 중: {file_path}")
        
        # 백업 생성
        backup_path = file_path + '.backup'
        os.rename(file_path, backup_path)
        
        fixed_lines = []
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"   ❌ 라인 {line_num} 제거: 잘못된 형식")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # 클래스 ID 수정
                    if class_id < 0:
                        print(f"   🔄 라인 {line_num}: 클래스 {class_id} → 0으로 수정")
                        class_id = 0
                    elif class_id > 2:
                        print(f"   🔄 라인 {line_num}: 클래스 {class_id} → 2로 수정")
                        class_id = 2
                    
                    # 좌표 범위 수정
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    w = max(0.001, min(1, w))  # 최소 크기 보장
                    h = max(0.001, min(1, h))
                    
                    fixed_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                    fixed_lines.append(fixed_line)
                    
                except ValueError:
                    print(f"   ❌ 라인 {line_num} 제거: 파싱 불가")
                    continue
        
        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
            if fixed_lines:  # 마지막에 개행 추가
                f.write('\n')
        
        print(f"   ✅ 수정 완료: {len(fixed_lines)}개 라인 저장")

def main():
    dataset_path = "/Users/hyunwoo/Desktop/DroneProject/yolov11_project/dataset"
    
    print("🔍 YOLO 라벨 데이터 검증 시작...")
    
    issues, problematic_files = validate_yolo_labels(dataset_path)
    
    print(f"\n📋 검증 결과:")
    print(f"   총 문제점: {len(issues)}개")
    print(f"   문제 파일: {len(problematic_files)}개")
    
    if issues:
        print(f"\n⚠️  발견된 문제점들:")
        for issue in issues[:20]:  # 처음 20개만 출력
            print(f"   {issue}")
        
        if len(issues) > 20:
            print(f"   ... 외 {len(issues)-20}개 추가 문제")
    
    if problematic_files:
        print(f"\n🔧 문제가 있는 파일 수정을 시작합니다...")
        
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            fix_label_files(problematic_files)
            print(f"\n✅ 수정 완료! 원본은 .backup 파일로 저장되었습니다.")
            
            # 재검증
            print(f"\n🔄 재검증 중...")
            issues, problematic_files = validate_yolo_labels(dataset_path)
            if not issues:
                print("✅ 모든 문제가 해결되었습니다!")
            else:
                print(f"⚠️  {len(issues)}개의 문제가 여전히 남아있습니다.")
    else:
        print("✅ 라벨 데이터에 문제가 없습니다!")

if __name__ == "__main__":
    main()