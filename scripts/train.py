"""
YOLOv11 통합 학습 스크립트
- 디바이스 자동 감지
- 데이터 자동 분할
- W&B 시각화 (선택적)
- 하이퍼파라미터 최적화
"""

import os
import sys
import yaml
import wandb
import argparse
from pathlib import Path
from ultralytics import YOLO

# utils 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    detect_device,
    get_recommended_settings,
    WandBVisualizer,
    log_training_samples,
    create_data_yaml
)


def train_yolo(args):
    """
    통합 YOLOv11 학습 함수

    Args:
        args: 명령행 인자
    """
    print("\n" + "=" * 70)
    print("                YOLOv11 드론 이미지 탐지 학습")
    print("=" * 70)

    # 1. 디바이스 자동 감지
    print("\n🔍 디바이스 자동 감지...")
    device_type, device_info = detect_device()
    settings = get_recommended_settings(device_type, device_info)

    print(f"✅ 감지된 디바이스: {device_info['type']}")
    print(f"📱 디바이스: {settings['device']}")
    print(f"📦 최적 배치 크기: {settings['batch_size']}")
    print(f"⚙️  워커 수: {settings['workers']}")

    # 2. 설정 파일 로드
    if args.config:
        config_file = Path(args.config)
    else:
        # 디바이스에 맞는 설정 파일 자동 선택
        config_file = Path(__file__).parent.parent / 'configs' / settings['config_file']

    print(f"📋 설정 파일: {config_file.name}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 프로젝트 경로 설정
    project_dir = Path(__file__).parent.parent
    dataset_dir = project_dir / 'dataset'  # 이미 변환된 YOLO 데이터

    # dataset이 없으면 에러
    if not dataset_dir.exists():
        print("❌ dataset 폴더가 없습니다!")
        print("먼저 다음 명령을 실행하세요:")
        print("  1. 폴더 이름 변경: mv dataset_perfect dataset")
        print("  2. 또는 변환 실행: python utils/perfect_yolo_converter.py")
        return

    # 4. 데이터 준비
    data_yaml_path = project_dir / 'configs' / 'data.yaml'

    # 데이터가 이미 분할되어 있는지 확인
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    test_dir = dataset_dir / 'test'

    if not val_dir.exists() or args.force_split:
        print("\n📂 데이터 분할 필요...")

        # train 폴더만 있는 경우 분할 실행
        if train_dir.exists():
            images_dir = train_dir / 'images'
            labels_dir = train_dir / 'labels'

            # 데이터 확인
            image_files = list(images_dir.glob('*.jpg'))
            label_files = list(labels_dir.glob('*.txt'))

            if len(image_files) == 0:
                print("❌ 에러: dataset/train/images/ 폴더에 이미지가 없습니다!")
                return

            print(f"📊 Train 데이터: 이미지 {len(image_files)}개, 라벨 {len(label_files)}개")

            # Val, Test 폴더 존재 확인
            val_dir = dataset_dir / "val"
            test_dir = dataset_dir / "test"

            if not val_dir.exists() or not test_dir.exists():
                print("❌ 에러: val 또는 test 폴더가 없습니다!")
                print("perfect_yolo_converter.py를 먼저 실행하여 데이터를 변환하고 분할하세요.")
                return

            val_images = len(list((val_dir / "images").glob('*.jpg')))
            val_labels = len(list((val_dir / "labels").glob('*.txt')))
            test_images = len(list((test_dir / "images").glob('*.jpg')))
            test_labels = len(list((test_dir / "labels").glob('*.txt')))

            print(f"📊 Val 데이터: 이미지 {val_images}개, 라벨 {val_labels}개")
            print(f"📊 Test 데이터: 이미지 {test_images}개, 라벨 {test_labels}개")
        else:
            print("❌ dataset/train 폴더가 없습니다!")
            return

    # 5. 데이터셋 검증
    if args.validate_data:
        print("\n🔍 데이터셋 검증 중...")
        for split in ['train', 'val', 'test']:
            split_path = dataset_dir / split
            if split_path.exists():
                img_count = len(list((split_path / 'images').glob('*')))
                label_count = len(list((split_path / 'labels').glob('*')))
                print(f"  {split}: 이미지 {img_count}개, 라벨 {label_count}개")
            else:
                print(f"  {split}: 폴더 없음")

    # 6. W&B 초기화 (선택적)
    if args.use_wandb:
        print("\n📊 W&B 초기화...")

        # W&B 설정 로드
        wandb_config_path = project_dir / 'configs' / 'wandb_config.yaml'
        with open(wandb_config_path, 'r') as f:
            wandb_config = yaml.safe_load(f)

        # 실행 이름 생성
        run_name = args.wandb_name or f"yolov11_{args.model.split('.')[0]}_{device_type}_{config.get('epochs', 100)}ep"

        # YOLO 내장 W&B 통합 사용 (원래 작동했던 방식)
        import os
        os.environ['WANDB_PROJECT'] = args.wandb_project or wandb_config['project']
        os.environ['WANDB_NAME'] = run_name
        if wandb_config.get('entity'):
            os.environ['WANDB_ENTITY'] = wandb_config['entity']

        print(f"✅ W&B 프로젝트: {args.wandb_project or wandb_config['project']}")
        print(f"📝 실행 이름: {run_name}")

        # 학습 후 샘플 로깅을 위해 변수 저장
        wandb_project_name = args.wandb_project or wandb_config['project']

    # 7. 모델 초기화 및 학습
    print(f"\n🚀 YOLOv11 모델 초기화: {args.model}")
    # 로컬 모델 파일 경로 확인
    model_path = project_dir / 'models' / args.model

    if model_path.exists():
        print(f"✅ 로컬 모델 사용: {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"⚠️ 로컬 모델이 없습니다. pretrained 모델을 다운로드합니다...")
        model = YOLO(args.model)  # 자동 다운로드

    print("\n" + "=" * 70)
    print("                    학습 시작")
    print("=" * 70)

    # 학습 파라미터 출력
    print(f"📋 주요 설정:")
    print(f"   에폭: {config.get('epochs', args.epochs)}")
    print(f"   배치: {args.batch or settings['batch_size']}")
    print(f"   이미지 크기: {config.get('imgsz', args.imgsz)}")
    print(f"   학습률: {config.get('lr0', 0.01)}")
    print(f"   옵티마이저: {config.get('optimizer', 'AdamW')}")

    if args.use_wandb:
        print(f"   W&B: 활성화 ({run_name})")


    # 학습 실행
    results = model.train(
        # 기본 설정
        data=str(data_yaml_path),
        epochs=config.get('epochs', args.epochs),
        imgsz=config.get('imgsz', args.imgsz),
        batch=args.batch or settings['batch_size'],
        device=args.device or settings['device'],
        workers=args.workers or settings['workers'],
        amp=settings['amp'] if args.amp is None else args.amp,
        cache=args.cache if args.cache is not None else settings.get('cache', 'disk'),
        project=str(project_dir / 'runs' / 'train'),
        name=args.exp_name or f"{device_type}_exp",
        exist_ok=args.exist_ok,
        pretrained=args.pretrained,
        verbose=args.verbose,
        seed=args.seed,

        # 옵티마이저 설정
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),

        # Warmup 설정
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),

        # 손실 가중치
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        label_smoothing=config.get('label_smoothing', 0.0),

        # 데이터 증강
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.1),
        copy_paste=config.get('copy_paste', 0.1),
        degrees=config.get('degrees', 15.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        shear=config.get('shear', 2.0),
        perspective=config.get('perspective', 0.001),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),

        # 기타 설정
        patience=config.get('patience', args.patience),
        save_period=config.get('save_period', -1),
        close_mosaic=config.get('close_mosaic', 10),
        cos_lr=config.get('cos_lr', False),
        resume=args.resume,
        fraction=config.get('fraction', 1.0),
        val=config.get('val', True),
        plots=config.get('plots', True),
    )

    # 8. 학습 완료 후 처리
    print("\n" + "=" * 70)
    print("                    학습 완료!")
    print("=" * 70)

    # 결과 경로
    exp_dir = project_dir / 'runs' / 'train' / (args.exp_name or f"{device_type}_exp")
    best_model_path = exp_dir / 'weights' / 'best.pt'
    last_model_path = exp_dir / 'weights' / 'last.pt'

    print(f"📁 결과 위치: {exp_dir}")
    print(f"🏆 최고 성능 모델: {best_model_path}")
    print(f"📊 마지막 모델: {last_model_path}")

    # 9. W&B 시각화 (선택적) - YOLO 내장 W&B 사용
    if args.use_wandb and best_model_path.exists():
        print("\n🎨 W&B 추가 시각화 중...")

        try:
            # W&B가 이미 YOLO에 의해 초기화되었으므로 추가 로깅만 수행
            if wandb.run is not None:
                # 최고 성능 모델로 시각화
                best_model = YOLO(str(best_model_path))
                visualizer = WandBVisualizer(args.class_names)

                if args.log_validation:
                    print("   📊 검증 결과 시각화...")
                    visualizer.log_validation_images(best_model, dataset_dir, num_images=args.num_val_images)

                if args.log_test:
                    print("   🎯 테스트 결과 시각화...")
                    visualizer.log_test_results(best_model, dataset_dir, num_images=args.num_test_images)

                # 모델 아티팩트 저장
                if args.save_model_artifact:
                    model_artifact = wandb.Artifact(
                        f"model_{wandb.run.id}",
                        type="model",
                        description=f"Best YOLOv11 model trained on {device_type}"
                    )
                    model_artifact.add_file(str(best_model_path))
                    wandb.log_artifact(model_artifact)
                    print("   💾 모델 아티팩트 저장 완료")

                print("✅ W&B 추가 시각화 완료!")
            else:
                print("⚠️  W&B 세션을 찾을 수 없습니다.")
        except Exception as e:
            print(f"⚠️  W&B 시각화 중 오류: {e}")

        # W&B 종료 (YOLO가 자동 관리하므로 조건부)
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass

    # 10. 최종 안내
    print(f"\n🎉 학습 완료!")
    if args.use_wandb:
        print(f"📊 W&B 대시보드에서 결과를 확인하세요")

    # 디바이스별 팁
    if device_type == 'mps':
        print("\n💡 Mac 사용자 팁:")
        print("   - Activity Monitor에서 GPU 사용률 확인")
        print("   - 메모리 부족 시 batch_size 감소")
    elif device_type == 'cuda':
        print("\n💡 GPU 사용자 팁:")
        print("   - nvidia-smi로 GPU 모니터링")
        print("   - 더 큰 모델 사용 고려 (yolov11s, yolov11m)")

    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='YOLOv11 통합 학습 스크립트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 기본 설정
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='YOLOv11 모델 (n/s/m/l/x.pt)')
    parser.add_argument('--config', type=str,
                       help='하이퍼파라미터 설정 파일 (자동 감지 시 생략 가능)')
    parser.add_argument('--exp-name', type=str,
                       help='실험 이름 (자동 생성)')

    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에폭 수')
    parser.add_argument('--batch', type=int,
                       help='배치 크기 (자동 감지 시 생략 가능)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='입력 이미지 크기')
    parser.add_argument('--device', type=str,
                       help='디바이스 (자동 감지 시 생략 가능)')
    parser.add_argument('--workers', type=int,
                       help='데이터 로더 워커 수 (자동 감지)')
    parser.add_argument('--patience', type=int, default=50,
                       help='조기 종료 patience')

    # 데이터 설정
    parser.add_argument('--class-names', nargs='+',
                       default=['crack', 'efflorescence', 'spalling'],
                       help='클래스 이름 리스트')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='학습 데이터 비율')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='검증 데이터 비율')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='테스트 데이터 비율')
    parser.add_argument('--force-split', action='store_true',
                       help='데이터 강제 재분할')
    parser.add_argument('--validate-data', action='store_true',
                       help='학습 전 데이터 검증')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')

    # W&B 설정
    parser.add_argument('--use-wandb', action='store_true',
                       help='W&B 사용')
    parser.add_argument('--wandb-project', type=str,
                       help='W&B 프로젝트 이름')
    parser.add_argument('--wandb-name', type=str,
                       help='W&B 실행 이름')
    parser.add_argument('--log-samples', action='store_true', default=True,
                       help='학습 샘플 로깅')
    parser.add_argument('--log-validation', action='store_true', default=True,
                       help='검증 결과 로깅')
    parser.add_argument('--log-test', action='store_true', default=True,
                       help='테스트 결과 로깅')
    parser.add_argument('--num-samples', type=int, default=15,
                       help='로깅할 학습 샘플 수')
    parser.add_argument('--num-val-images', type=int, default=10,
                       help='로깅할 검증 이미지 수')
    parser.add_argument('--num-test-images', type=int, default=20,
                       help='로깅할 테스트 이미지 수')
    parser.add_argument('--save-model-artifact', action='store_true', default=True,
                       help='W&B에 모델 아티팩트 저장')

    # 기타 옵션
    parser.add_argument('--amp', type=bool,
                       help='자동 혼합 정밀도 (자동 감지)')
    parser.add_argument('--cache', type=str, default=None,
                       help='이미지 캐싱 (True/False/disk)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='사전 학습 가중치 사용')
    parser.add_argument('--resume', action='store_true',
                       help='이전 학습 재개')
    parser.add_argument('--exist-ok', action='store_true', default=True,
                       help='기존 실험 덮어쓰기')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='상세 출력')

    args = parser.parse_args()

    # 학습 실행
    train_yolo(args)


if __name__ == '__main__':
    main()