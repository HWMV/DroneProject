#!/usr/bin/env python3
"""
바운딩 박스 범위 수정 스크립트
- 0-1 범위를 벗어나는 바운딩 박스를 수정
- box_loss=0 문제 해결을 위함
"""

import os
import shutil
from pathlib import Path
import numpy as np

def fix_bbox_coordinates(label_dir, backup=True, method='clip'):
    """
    바운딩 박스 좌표 수정

    Args:
        label_dir: 라벨 파일들이 있는 디렉토리
        backup: 백업 생성 여부
        method: 'clip' (범위 조정) 또는 'remove' (범위 초과 박스 제거)
    """
    label_dir = Path(label_dir)

    if backup:
        backup_dir = label_dir.parent / f"{label_dir.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(label_dir, backup_dir)
        print(f"📁 백업 생성: {backup_dir}")

    stats = {
        'total_files': 0,
        'modified_files': 0,
        'total_boxes': 0,
        'fixed_boxes': 0,
        'removed_boxes': 0,
        'issues_found': {
            'negative_left': 0,
            'over_right': 0,
            'negative_top': 0,
            'over_bottom': 0
        }
    }

    label_files = list(label_dir.glob('*.txt'))

    for label_file in label_files:
        stats['total_files'] += 1

        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            file_modified = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    modified_lines.append(line)
                    continue

                try:
                    cls_id, x, y, w, h = map(float, parts)
                    stats['total_boxes'] += 1

                    # 바운딩 박스 경계 계산
                    left = x - w/2
                    right = x + w/2
                    top = y - h/2
                    bottom = y + h/2

                    # 범위 초과 검사
                    needs_fix = False
                    eps = 1e-10

                    if left < -eps:
                        stats['issues_found']['negative_left'] += 1
                        needs_fix = True
                    if right > 1.0 + eps:
                        stats['issues_found']['over_right'] += 1
                        needs_fix = True
                    if top < -eps:
                        stats['issues_found']['negative_top'] += 1
                        needs_fix = True
                    if bottom > 1.0 + eps:
                        stats['issues_found']['over_bottom'] += 1
                        needs_fix = True

                    if needs_fix:
                        if method == 'clip':
                            # 클리핑: 범위를 0-1로 강제 조정
                            left = max(0.0, left)
                            right = min(1.0, right)
                            top = max(0.0, top)
                            bottom = min(1.0, bottom)

                            # 클리핑 후 새로운 center와 size 계산
                            new_w = right - left
                            new_h = bottom - top
                            new_x = left + new_w/2
                            new_y = top + new_h/2

                            # 너무 작아진 박스는 제거
                            if new_w < 0.001 or new_h < 0.001:
                                stats['removed_boxes'] += 1
                                file_modified = True
                                continue

                            modified_lines.append(f"{int(cls_id)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")
                            stats['fixed_boxes'] += 1
                            file_modified = True

                        elif method == 'remove':
                            # 제거: 범위 초과 박스 삭제
                            stats['removed_boxes'] += 1
                            file_modified = True
                            continue
                    else:
                        # 정상 박스는 그대로 유지
                        modified_lines.append(line)

                except ValueError:
                    # 파싱 오류가 있는 라인은 그대로 유지
                    modified_lines.append(line)
                    continue

            # 파일이 수정되었으면 저장
            if file_modified:
                stats['modified_files'] += 1
                with open(label_file, 'w') as f:
                    for line in modified_lines:
                        f.write(line + '\\n')

        except Exception as e:
            print(f"❌ 파일 처리 오류: {label_file} - {e}")
            continue

    return stats

def print_stats(stats):
    """통계 출력"""
    print(f"\\n📊 바운딩 박스 수정 결과:")
    print(f"총 파일: {stats['total_files']}개")
    print(f"수정된 파일: {stats['modified_files']}개")
    print(f"총 박스: {stats['total_boxes']}개")
    print(f"수정된 박스: {stats['fixed_boxes']}개")
    print(f"제거된 박스: {stats['removed_boxes']}개")

    print(f"\\n🚨 발견된 문제:")
    for issue, count in stats['issues_found'].items():
        if count > 0:
            print(f"❌ {issue}: {count}개")
        else:
            print(f"✅ {issue}: 없음")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='바운딩 박스 범위 수정')
    parser.add_argument('--label-dir', required=True, help='라벨 디렉토리 경로')
    parser.add_argument('--method', choices=['clip', 'remove'], default='clip',
                       help='수정 방법: clip (범위 조정) 또는 remove (제거)')
    parser.add_argument('--no-backup', action='store_true', help='백업 생성 안함')

    args = parser.parse_args()

    stats = fix_bbox_coordinates(
        label_dir=args.label_dir,
        backup=not args.no_backup,
        method=args.method
    )

    print_stats(stats)