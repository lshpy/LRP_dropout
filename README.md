# 🧠 Suppressive Relevance 기반 Dropout 및 Amplification 실험 프레임워크

이 프로젝트는 LRP(Layer-wise Relevance Propagation) 기반 suppressive relevance 정보를 활용하여,  
해석 가능한 dropout/amplification 전략을 학습 과정에 직접 반영하는 딥러닝 실험 프레임워크입니다.

총 **18가지 실험 조건**을 구성하여 Pixel / Patch / Channel 단위의 다양한 dropout/amplify 전략을 실험할 수 있습니다.

---

## 📁 디렉토리 구조

```
suppressive_dropout_experiment/
├── train.py              # 단일 실험 실행용 메인 스크립트
├── run_all_experiments.py # 18개 조건 전체 자동 실행 스크립트
├── model.py              # ResNet18 기반 모델 (feature hook 포함)
├── strategy/             # dropout/amplify 전략 구현 모듈
│   ├── __init__.py
│   ├── random.py         # 무작위 dropout 전략
│   ├── suppressive.py    # LRP 기반 suppressive dropout
│   ├── gradcam.py        # Grad-CAM 기반 amplify 전략
│   ├── hybrid.py         # hybrid dropout/amplify 전략
│   ├── mixed.py          # dropout + amplify 혼합 전략
│   └── recovery.py       # dropout된 feature 복원 loss 전략
├── utils/                # 보조 유틸리티 모듈
│   ├── lrp.py            # suppressive relevance 계산
│   ├── gradcam.py        # Grad-CAM saliency 계산
│   └── visualization.py  # dropout/amplify mask 시각화
├── results/              # 실험 결과 저장 (CSV 등)
└── visualizations/       # dropout/amplify mask 이미지 저장
```

---

## 🚀 사용 방법

### 1. CIFAR-10 단일 전략 실험 (예: hybrid_drop @ patch)
```bash
python train.py --strategy hybrid_drop --unit patch
```

### 2. 전체 18조건 자동 실행
```bash
python run_all_experiments.py
```

### 3. 결과 확인

---

## 🧪 실험 조건 구성

| 단위    | 전략 종류           | 예시 조건명                    |
|---------|---------------------|-------------------------------|
| Pixel   | Suppressive Dropout | suppressive @ pixel           |
| Patch   | Hybrid Amplify      | hybrid_amp @ patch            |
| Channel | Mixed               | mixed @ channel               |

총 3단위 × 6전략 + baseline = **18조건**

---

## 📌 참고

- 본 실험은 Phase 1 (CIFAR-10) 구조 점검 이후, Phase 2 (Chest X-ray) 해석성 평가로 확장됩니다.
- Grad-CAM과 LRP는 [Captum](https://captum.ai) 또는 custom backprop으로 구현 가능합니다.

---

📬 문의: leesh4660@gmail.com 
