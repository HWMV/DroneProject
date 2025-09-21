import os
from PIL import Image
from pathlib import Path
import shutil

def convert_tiff_to_jpg(input_folder, output_folder, quality=95, preserve_metadata=True):
    """
    TIFF 파일들을 JPG로 변환하는 함수
    
    Args:
        input_folder: TIFF 파일들이 있는 입력 폴더
        output_folder: JPG 파일들을 저장할 출력 폴더
        quality: JPG 품질 (1-100, 95 권장)
        preserve_metadata: GPS 등 메타데이터 보존 여부
    """
    
    # 출력 폴더 생성
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    error_count = 0
    
    # TIFF 파일들 처리
    for tiff_file in Path(input_folder).glob("*.tiff"):
        try:
            # TIFF 파일 열기
            with Image.open(tiff_file) as img:
                # RGB로 변환 (JPG는 RGBA 지원 안함)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # 출력 파일 경로 (확장자만 변경)
                jpg_filename = tiff_file.stem + ".jpg"
                output_path = Path(output_folder) / jpg_filename
                
                # 메타데이터 보존 옵션
                save_kwargs = {
                    'format': 'JPEG',
                    'quality': quality,
                    'optimize': True
                }
                
                if preserve_metadata:
                    # EXIF 데이터 보존
                    if hasattr(img, '_getexif') and img._getexif():
                        save_kwargs['exif'] = img.info.get('exif', b'')
                
                # JPG로 저장
                img.save(output_path, **save_kwargs)
                
                converted_count += 1
                print(f"변환 완료: {tiff_file.name} → {jpg_filename}")
                
        except Exception as e:
            error_count += 1
            print(f"변환 실패: {tiff_file.name} - 오류: {str(e)}")
    
    print(f"\n=== 변환 결과 ===")
    print(f"성공: {converted_count}개")
    print(f"실패: {error_count}개")
    print(f"저장 위치: {output_folder}")

def batch_convert_multiple_folders(base_input_folder, base_output_folder, quality=95):
    """
    여러 TIFF 폴더를 한번에 JPG로 변환하는 함수
    """
    
    base_input_path = Path(base_input_folder)
    
    for subfolder in base_input_path.iterdir():
        if subfolder.is_dir():
            print(f"\n처리 중: {subfolder.name}")
            
            # 출력 폴더 경로
            output_folder = Path(base_output_folder) / subfolder.name
            
            # 변환 실행
            convert_tiff_to_jpg(
                str(subfolder),
                str(output_folder),
                quality=quality
            )

def compare_file_sizes(tiff_folder, jpg_folder):
    """
    TIFF와 JPG 용량 비교 함수
    """
    
    tiff_total_size = 0
    jpg_total_size = 0
    
    # TIFF 폴더 용량 계산
    for tiff_file in Path(tiff_folder).glob("*.tiff"):
        tiff_total_size += tiff_file.stat().st_size
    
    # JPG 폴더 용량 계산  
    for jpg_file in Path(jpg_folder).glob("*.jpg"):
        jpg_total_size += jpg_file.stat().st_size
    
    # 결과 출력
    tiff_gb = tiff_total_size / (1024**3)
    jpg_gb = jpg_total_size / (1024**3)
    reduction_rate = (1 - jpg_total_size/tiff_total_size) * 100
    
    print(f"\n=== 용량 비교 ===")
    print(f"TIFF 총 용량: {tiff_gb:.2f} GB")
    print(f"JPG 총 용량: {jpg_gb:.2f} GB") 
    print(f"용량 절약률: {reduction_rate:.1f}%")

# 사용 예시
if __name__ == "__main__":
    # === 단일 폴더 변환 ===
    input_folder = "RawData/원천데이터/박리/콘크리트_박리_원천_19" 
    # "RawData/원천데이터/균열/콘크리트_콘크리트균열_원천_38"
    output_folder = "RawData/원천데이터_JPG/박리/콘크리트_박리_원천_19"
    
    print("TIFF → JPG 변환 시작...")
    convert_tiff_to_jpg(
        input_folder, 
        output_folder, 
        quality=95,          # 품질 95% (권장)
        preserve_metadata=True  # GPS 등 메타데이터 보존
    )
    
    # 용량 비교
    compare_file_sizes(input_folder, output_folder)
    
    print("\n" + "="*50 + "\n")
    
    # === 여러 폴더 일괄 변환 ===
    # base_input_folder = "RawData/원천데이터/균열"  # 여러 하위폴더 포함
    # base_output_folder = "RawData/원천데이터_JPG/균열"
    # 
    # print("일괄 변환 시작...")
    # batch_convert_multiple_folders(base_input_folder, base_output_folder, quality=95)