# 🚁 YOLOv11 Drone Detection Project

드론으로 촬영한 인프라 결함 탐지를 위한 YOLOv11 기반 객체 탐지 시스템

## 📋 프로젝트 개요

- **목적**: 드론 이미지에서 콘크리트 구조물의 결함 자동 탐지
- **모델**: YOLOv11n (경량화 버전)
- **클래스**: 3가지 결함 유형
  - `ConcreteCrack` (균열)
  - `Efflorescence` (백태)
  - `Spalling` (박리)

## 🎯 주요 기능

- **다중 플랫폼 지원**: Mac MPS, CUDA GPU, CPU
- **실험 추적**: Weights & Biases (W&B) 통합
- **자동화된 학습 파이프라인**: 디바이스 자동 감지 및 최적화
- **데이터 전처리**: JSON to YOLO 형식 변환 도구
- **시각화**: 학습 샘플 및 예측 결과 W&B 로깅

## 🚀 빠른 시작

### 1. 환경 설정
```bash
git clone git@github.com:HWMV/DroneProject.git
cd DroneProject
pip install -r requirements.txt
```

### 2. 학습 실행
```bash
# CPU 학습 (기본)
./train_cpu.sh

# 에폭 수 변경
./train_cpu.sh --epochs 50

# 배치 크기 변경
./train_cpu.sh --batch 16
```

### 3. 추가 학습 (체크포인트에서 이어서)
```bash
# 기본: best.pt에서 30 에폭 추가
./continue_train.sh

# 사용자 정의
./continue_train.sh runs/train/exp1/weights/best.pt my_experiment 50
```

## 📁 프로젝트 구조

```
DroneProject/
├── configs/                 # 설정 파일들
│   ├── data.yaml            # 데이터셋 설정
│   ├── hyp_mac.yaml         # Mac 최적화 하이퍼파라미터
│   ├── hyp_cuda.yaml        # CUDA GPU 설정
│   └── wandb_config.yaml    # W&B 실험 추적 설정
├── scripts/
│   └── train.py             # 통합 학습 스크립트
├── utils/                   # 유틸리티 도구들
│   ├── dataUtils/           # 데이터 전처리 도구
│   ├── detect_device.py     # 디바이스 자동 감지
│   └── wandb_visualize.py   # W&B 시각화
├── models/
│   └── yolo11n.pt          # 베이스 YOLOv11n 모델
├── train_cpu.sh            # CPU 학습 스크립트
├── continue_train.sh       # 추가 학습 스크립트
└── requirements.txt        # 의존성 패키지
```

## ⚙️ 설정 파일

### hyp_mac.yaml - Mac 최적화 설정
```yaml
device: cpu  # 또는 mps
batch_size: 8
workers: 0   # MPS에서는 0 권장
epochs: 30
lr0: 0.01
optimizer: 'SGD'
```

### data.yaml - 데이터셋 설정
```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

names:
  0: ConcreteCrack
  1: Efflorescene
  2: Spalling
```

## 📊 성능 지표

학습된 모델의 성능은 W&B 대시보드에서 실시간으로 모니터링할 수 있습니다:

- **mAP@0.5**: 목표 10% 이상
- **Precision/Recall**: 클래스별 세부 성능
- **Loss Curves**: train/val loss 추적
- **Learning Rate**: 최적화 과정 시각화

## 🛠️ 데이터 전처리

포함된 유틸리티 도구들:

1. **JSON to YOLO 변환**: `utils/dataUtils/2.perfect_yolo_converter.py`
2. **이미지-라벨 매칭**: `utils/dataUtils/3.image_label_match.py`
3. **클래스 ID 검증**: `utils/dataUtils/4.check_class_ids.py`
4. **바운딩 박스 범위 수정**: `utils/dataUtils/5.fix_bbox_ranges.py`

## 💻 시스템 요구사항

- **Python**: 3.12+
- **PyTorch**: 2.8.0+
- **메모리**: 최소 8GB RAM
- **저장공간**: 모델 학습 시 ~500MB

### 권장 환경
- **Mac**: M1/M2/M3 (MPS 지원)
- **GPU**: NVIDIA RTX 시리즈 (CUDA 11.8+)
- **CPU**: Intel/AMD 멀티코어

## 🔧 문제 해결

### Mac MPS 이슈
```bash
# MPS 대신 CPU 사용
./train_cpu.sh --device cpu

# Workers 수 조정
./train_cpu.sh --workers 0
```

### 메모리 부족
```bash
# 배치 크기 감소
./train_cpu.sh --batch 4

# 캐시 비활성화
./train_cpu.sh --cache False
```

## 📈 실험 추적

W&B 대시보드에서 확인 가능한 정보:
- 실시간 loss/mAP 그래프
- 학습 샘플 시각화
- 하이퍼파라미터 비교
- 모델 아티팩트 관리

## 🤝 기여하기

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🚁 드론 이미지 처리 특화 기능

- **고해상도 이미지 지원**: 드론에서 촬영된 고해상도 이미지 처리
- **다양한 촬영 각도**: 여러 시점에서의 결함 탐지
- **실시간 추론**: 경량화된 YOLOv11n으로 빠른 처리
- **결함 유형별 최적화**: 균열, 백태, 박리 각각에 특화된 탐지

---

🤖 **Generated with [Claude Code](https://claude.ai/code)**