"""
디바이스 자동 감지 및 최적 설정 스크립트
Mac(MPS), CUDA GPU, CPU를 자동으로 감지하여 적절한 설정 파일 선택
"""

import torch
import platform
import subprocess
import sys
from pathlib import Path


def detect_device():
    """
    사용 가능한 디바이스를 자동으로 감지

    Returns:
        device_type: 'mps', 'cuda', 'cpu'
        device_info: 디바이스 상세 정보
    """
    device_info = {}

    # 시스템 정보
    system = platform.system()
    device_info['system'] = system
    device_info['platform'] = platform.platform()
    device_info['processor'] = platform.processor()

    # Mac (Apple Silicon) 체크
    if system == 'Darwin':  # macOS
        try:
            # Apple Silicon 확인
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            device_info['cpu'] = cpu_info

            if 'Apple' in cpu_info and torch.backends.mps.is_available():
                device_info['device'] = 'mps'
                device_info['type'] = 'Apple Silicon GPU (Metal Performance Shaders)'
                return 'mps', device_info
        except:
            pass

    # CUDA GPU 체크
    if torch.cuda.is_available():
        device_info['device'] = 'cuda'
        device_info['cuda_version'] = torch.version.cuda
        device_info['gpu_count'] = torch.cuda.device_count()
        device_info['gpu_names'] = []
        device_info['gpu_memory'] = []

        for i in range(torch.cuda.device_count()):
            device_info['gpu_names'].append(torch.cuda.get_device_name(i))
            # GPU 메모리 정보 (GB 단위)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            device_info['gpu_memory'].append(f"{total_memory:.2f} GB")

        device_info['type'] = 'NVIDIA CUDA GPU'
        return 'cuda', device_info

    # CPU 폴백
    device_info['device'] = 'cpu'
    device_info['type'] = 'CPU Only'
    device_info['cpu_count'] = torch.get_num_threads()

    return 'cpu', device_info


def get_recommended_settings(device_type, device_info):
    """
    디바이스에 따른 권장 설정 반환

    Args:
        device_type: 'mps', 'cuda', 'cpu'
        device_info: 디바이스 정보

    Returns:
        settings: 권장 설정 딕셔너리
    """
    settings = {}

    if device_type == 'mps':
        # Mac MPS 설정
        settings['config_file'] = 'hyp_mac.yaml'
        settings['device'] = 'mps'
        settings['batch_size'] = 8
        settings['workers'] = 4
        settings['amp'] = False
        settings['cache'] = True
        settings['notes'] = 'Mac M1/M2/M3 최적화 설정'

    elif device_type == 'cuda':
        # CUDA GPU 설정
        settings['config_file'] = 'hyp_cuda.yaml'
        settings['device'] = '0' if device_info['gpu_count'] == 1 else '0,1'

        # GPU 메모리에 따른 배치 크기 조정
        if device_info.get('gpu_memory'):
            memory = float(device_info['gpu_memory'][0].split()[0])
            if memory < 8:
                settings['batch_size'] = 8
            elif memory < 12:
                settings['batch_size'] = 16
            elif memory < 16:
                settings['batch_size'] = 32
            elif memory < 24:
                settings['batch_size'] = 64
            else:
                settings['batch_size'] = 128
        else:
            settings['batch_size'] = 16

        settings['workers'] = 16
        settings['amp'] = True
        settings['cache'] = False
        settings['notes'] = f"NVIDIA GPU 최적화 ({device_info.get('gpu_names', ['Unknown'])[0]})"

    else:
        # CPU 설정
        settings['config_file'] = 'hyp.yaml'
        settings['device'] = 'cpu'
        settings['batch_size'] = 4
        settings['workers'] = 2
        settings['amp'] = False
        settings['cache'] = True
        settings['notes'] = 'CPU 전용 설정 (느림 주의)'

    return settings


def print_device_info(device_type, device_info, settings):
    """
    디바이스 정보와 권장 설정 출력
    """
    print("=" * 60)
    print("            디바이스 감지 결과")
    print("=" * 60)

    print(f"시스템: {device_info.get('system', 'Unknown')}")
    print(f"프로세서: {device_info.get('processor', 'Unknown')}")

    if device_type == 'mps':
        print(f"디바이스: {device_info['type']}")
        print(f"CPU: {device_info.get('cpu', 'Unknown')}")

    elif device_type == 'cuda':
        print(f"디바이스: {device_info['type']}")
        print(f"CUDA 버전: {device_info.get('cuda_version', 'Unknown')}")
        print(f"GPU 개수: {device_info.get('gpu_count', 0)}")
        for i, (name, mem) in enumerate(zip(device_info.get('gpu_names', []),
                                           device_info.get('gpu_memory', []))):
            print(f"  GPU {i}: {name} ({mem})")

    else:
        print(f"디바이스: {device_info['type']}")
        print(f"CPU 스레드: {device_info.get('cpu_count', 'Unknown')}")

    print("\n" + "=" * 60)
    print("            권장 설정")
    print("=" * 60)
    print(f"설정 파일: configs/{settings['config_file']}")
    print(f"디바이스: {settings['device']}")
    print(f"배치 크기: {settings['batch_size']}")
    print(f"워커 수: {settings['workers']}")
    print(f"AMP: {settings['amp']}")
    print(f"캐시: {settings['cache']}")
    print(f"비고: {settings['notes']}")

    print("\n" + "=" * 60)
    print("            사용 방법")
    print("=" * 60)
    print(f"cd scripts")
    print(f"python train.py --cfg ../configs/{settings['config_file']}")

    if device_type == 'mps':
        print("\n⚠️  Mac 사용 시 주의사항:")
        print("  - 첫 실행 시 Metal 컴파일로 인해 느릴 수 있음")
        print("  - 메모리 부족 시 batch_size를 4로 줄이세요")
        print("  - AMP는 지원되지 않습니다")

    elif device_type == 'cuda':
        print("\n✅ GPU 최적화 팁:")
        print("  - nvidia-smi로 GPU 사용률 모니터링")
        print("  - OOM 에러 시 batch_size 감소")
        print("  - 멀티 GPU 사용: --device 0,1,2,3")

    else:
        print("\n⚠️  CPU 사용 시 주의사항:")
        print("  - 학습 속도가 매우 느립니다")
        print("  - 작은 데이터셋으로 테스트 권장")
        print("  - 가능하면 GPU 환경 사용 권장")


def create_auto_config(device_type, settings):
    """
    자동 감지된 설정으로 auto_config.yaml 생성
    """
    config_path = Path('../configs/auto_detected.yaml')

    # 기본 설정 파일 읽기
    base_config_path = Path(f'../configs/{settings["config_file"]}')

    if base_config_path.exists():
        import shutil
        shutil.copy(base_config_path, config_path)
        print(f"\n✅ 자동 설정 파일 생성: configs/auto_detected.yaml")
        print("   python train.py --cfg ../configs/auto_detected.yaml")


def main():
    """메인 함수"""
    print("\n🔍 디바이스 자동 감지 중...")

    # 디바이스 감지
    device_type, device_info = detect_device()

    # 권장 설정 가져오기
    settings = get_recommended_settings(device_type, device_info)

    # 정보 출력
    print_device_info(device_type, device_info, settings)

    # 자동 설정 파일 생성
    create_auto_config(device_type, settings)

    return device_type, settings


if __name__ == '__main__':
    device_type, settings = main()