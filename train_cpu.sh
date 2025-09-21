#!/bin/bash
# CPU 학습 간편 실행 스크립트

echo "🚀 YOLOv11 CPU 학습 시작..."
echo "📊 W&B 프로젝트: YOLOv11-Drone-Detection"
echo "⚠️  CPU 학습은 시간이 오래 걸립니다"
echo ""

python scripts/train.py \
  --model yolo11n.pt \
  --config configs/hyp_mac.yaml \
  --device cpu \
  --batch 8 \
  --workers 4 \
  --cache False \
  --use-wandb \
  --wandb-project "YOLOv11-Drone-Detection" \
  --log-validation \
  --log-test \
  "$@"

echo ""
echo "🎉 학습 완료!"