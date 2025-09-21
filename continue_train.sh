#!/bin/bash
# 범용 추가 학습 스크립트
# 사용법: ./continue_train.sh [체크포인트] [실험이름] [에폭수]
# 예시: ./continue_train.sh runs/train/mps_exp/weights/best.pt exp_v2 50

# 기본값 설정
CHECKPOINT=${1:-"runs/train/mps_exp/weights/best.pt"}
EXP_NAME=${2:-"continued_$(date +%y%m%d_%H%M)"}
EPOCHS=${3:-30}

echo "=========================================="
echo "🚀 YOLOv11 추가 학습"
echo "=========================================="
echo "📂 체크포인트: $CHECKPOINT"
echo "🏷️  실험 이름: $EXP_NAME"
echo "🔄 에폭 수: $EPOCHS"
echo "=========================================="
echo ""

# 체크포인트 존재 확인
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT"
    exit 1
fi

# 학습 실행
python scripts/train.py \
  --model "$CHECKPOINT" \
  --config configs/hyp_mac.yaml \
  --device cpu \
  --batch 8 \
  --workers 0 \
  --cache False \
  --epochs "$EPOCHS" \
  --use-wandb \
  --wandb-project "YOLOv11-Drone-Detection" \
  --exp-name "$EXP_NAME" \
  --log-validation \
  --log-test

echo ""
echo "🎉 학습 완료!"
echo "📁 결과 위치: runs/train/$EXP_NAME/"