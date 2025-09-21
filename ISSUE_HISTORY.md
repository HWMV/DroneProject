# 🐛 Issue History & Solutions

프로젝트 개발 과정에서 발생한 주요 이슈들과 해결 방법을 기록합니다.

## 📋 목차

1. [데이터셋 관련 이슈](#-데이터셋-관련-이슈)
2. [Mac MPS 학습 문제](#-mac-mps-학습-문제)
3. [W&B 시각화 문제](#-wb-시각화-문제)
4. [라벨 형식 문제](#-라벨-형식-문제)
5. [학습 재개 이슈](#-학습-재개-이슈)

---

## 🗂️ 데이터셋 관련 이슈

### Issue 1: JSON to YOLO 변환 문제
**문제**: 원본 JSON 라벨을 YOLO 형식으로 변환 시 좌표 정규화 오류
```
- 바운딩 박스 좌표가 이미지 크기를 벗어남
- 클래스 ID 불일치 문제
```

**해결책**: `utils/dataUtils/2.perfect_yolo_converter.py` 개선
```python
# 정규화 좌표 확인 및 클램핑
x_center = max(0, min(1, x_center))
y_center = max(0, min(1, y_center))
width = max(0, min(1, width))
height = max(0, min(1, height))
```

### Issue 2: 이미지-라벨 매칭 불일치
**문제**: 일부 이미지에 대응하는 라벨 파일이 없거나 반대의 경우
```
Found 7996 images but 7995 labels
```

**해결책**: `utils/dataUtils/3.image_label_match.py`로 자동 매칭 및 정리

---

## 🖥️ Mac MPS 학습 문제

### Issue 3: MPS 학습 시 NaN 발생
**문제**: Mac MPS로 학습 시 loss가 NaN으로 발산
```
epoch 1: loss=nan, box_loss=nan, cls_loss=nan
```

**원인**:
- MPS의 부동소수점 연산 불안정성
- 잘못된 worker 설정
- 캐시 설정 문제

**해결책**:
```yaml
# hyp_mac.yaml 설정
device: cpu  # mps 대신 cpu 사용
workers: 0   # MPS에서는 0 필수
amp: false   # AMP 비활성화
cache: 'disk'  # 메모리 캐시 대신 디스크 사용
optimizer: 'SGD'  # AdamW 대신 SGD 사용
```

### Issue 4: MPS 메모리 부족
**문제**: 큰 배치 크기 사용 시 메모리 부족
```
RuntimeError: MPS backend out of memory
```

**해결책**:
```bash
# 배치 크기 감소
--batch 4  # 기본 8에서 4로 감소
--workers 0
```

---

## 📊 W&B 시각화 문제

### Issue 5: W&B 메트릭이 표시되지 않음
**문제**: W&B 대시보드에 Charts, Media, System만 보이고 상세 메트릭 없음

**원인 분석**:
1. YOLO 내장 W&B 통합과 수동 `wandb.init()` 충돌
2. 환경변수 설정 미적용
3. 프로젝트명이 경로로 표시되는 문제

**시도한 해결책들**:
```python
# 방법 1: 수동 wandb.init() 제거, 환경변수 사용
os.environ['WANDB_PROJECT'] = 'YOLOv11-Drone-Detection'
os.environ['WANDB_NAME'] = run_name

# 방법 2: model.train()에 직접 project 파라미터 전달
model.train(project='YOLOv11-Drone-Detection', ...)

# 방법 3: 수동 콜백으로 메트릭 로깅
def on_train_epoch_end(trainer):
    wandb.log(metrics, step=trainer.epoch)
```

**최종 해결**: YOLO 내장 W&B 통합 사용, 1 에폭 완료 후 메트릭 표시 확인

---

## 🏷️ 라벨 형식 문제

### Issue 6: 라벨 파일 개행 문자 오류
**문제**: 라벨 파일에 리터럴 `\n` 문자열이 포함됨
```
0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n
```

**발견**: 763개 파일에서 문제 발생

**해결책**: `fix_labels.py` 스크립트로 일괄 수정
```python
content = content.replace('\\n', '\n')
```

### Issue 7: 캐시 파일 손상
**문제**: 잘못된 라벨로 인한 캐시 파일 손상
```
Error loading dataset cache
```

**해결책**: 캐시 파일 삭제 후 재생성
```bash
find . -name "*.cache" -delete
```

---

## 🔄 학습 재개 이슈

### Issue 8: Resume 학습 실패
**문제**: 30 에폭 완료된 모델에서 추가 학습 시도 시 에러
```
AssertionError: training to 30 epochs is finished, nothing to resume.
```

**원인**: `--resume` 옵션은 중단된 학습만 재개 가능

**해결책**:
```bash
# resume 대신 새로운 학습으로 시작
python train.py --model best.pt --epochs 30  # resume 플래그 제거
```

### Issue 9: 체크포인트 선택 문제
**문제**: last.pt vs best.pt 선택 기준 불명확

**해결책**:
- **best.pt**: 검증 성능 최고 → 추가 학습에 적합
- **last.pt**: 마지막 에폭 → 중단된 학습 재개에 적합

---

## ⚙️ 설정 최적화

### 최종 권장 설정 (Mac)
```yaml
# hyp_mac.yaml
device: cpu
batch_size: 8
workers: 0
epochs: 30
lr0: 0.01
optimizer: 'SGD'
cache: 'disk'
amp: false
```

### GPU 사용자 권장 설정
```yaml
# hyp_cuda.yaml
device: 0
batch_size: 16
workers: 8
epochs: 100
lr0: 0.01
optimizer: 'AdamW'
cache: true
amp: true
```

---

## 📈 성능 개선 팁

1. **데이터 품질 체크**: 라벨 파일 형식 및 바운딩 박스 범위 검증
2. **하이퍼파라미터 조정**: 플랫폼별 최적화된 설정 사용
3. **메모리 관리**: 배치 크기 및 워커 수 조정
4. **캐시 활용**: 디스크 캐시로 안정성 확보
5. **실험 추적**: W&B로 체계적인 실험 관리

---

## 🔮 알려진 제한사항

1. **MPS 안정성**: Mac MPS는 아직 실험적 기능으로 CPU 사용 권장
2. **대용량 데이터셋**: 29GB+ 데이터셋은 Git에서 제외
3. **메모리 요구사항**: 최소 8GB RAM 필요
4. **W&B 동기화**: 네트워크 연결 필수

---

*마지막 업데이트: 2024-09-21*