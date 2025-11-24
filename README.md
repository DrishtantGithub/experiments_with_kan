KAN-Enhanced Deep Learning: Robust, Interpretable, and Efficient Models

This repository contains the full implementation, experiments, and analysis for our EE782 Advanced ML Final Project, where we build, extend, and analyze Kolmogorov–Arnold Networks (KANs) across multiple domains — vision, NLP, tabular regression, and toy function approximation — and benchmark them against classical architectures (MLP & CNN).

We further introduce a Residual KAN Head, perform extensive ablation, robustness, interpretability, and efficiency analyses, and provide a clean modular codebase suitable for further research.

🚀 Project Highlights
✔ Unified KAN Benchmarking Across Modalities

Toy Regression (sinusoid)

CIFAR-10 image classification

IMDB sentiment analysis

Housing & Energy tabular regression tasks

✔ Novel Architecture

We introduce a Residual KAN Head for CNNs:

combines linear skip connections

stabilizes spline curvature

improves robustness + efficiency

✔ Deep Experimental Suite

We perform:

Knot Ablations (1, 3, 5, 7 knots)

Spline Curvature Regularization Ablation

Noise Robustness Experiments

Low-Data Generalization

Efficiency (Params, MACs, Latency, Model Size)

Full Interpretability:

Spline visualization

Derivative smoothness

Knot importance

Activation Patterns

Locality measurements

✔ Reproducible Pipelines

Every experiment is runnable end-to-end using:

python -m src.train....
python -m src.analysis....
python -m src.robustness....

📂 Repository Structure
kan_project/
│
├── src/
│   ├── models/                 # MLP, CNN, KAN, Residual KAN
│   ├── train/                  # Training scripts for toy, CIFAR, NLP, tabular
│   ├── analysis/               # Interpretability, spline plots, locality, activations
│   ├── robustness/             # Noise & low-data robustness experiments
│   └── utils/                  # Data loaders, metrics, plotting helpers
│
├── results/                    # Saved trained models + experiment outputs
│   ├── toy_kan/
│   ├── cifar_relu/
│   ├── cifar_kan/
│   ├── cifar_residual/
│   ├── interpretability/
│   ├── robustness/
│   ├── tabular_housing/
│   ├── tabular_energy/
│   └── nlp_imdb/
│
├── paper/                      # LaTeX source for IEEE paper
│   ├── Images/
│   └── sections/
│
├── requirements.txt
├── environment.yml
└── README.md

🧪 How to Run Experiments
1. Create Environment
conda create -n kan python=3.10
pip install -r requirements.txt

🎯 Training Pipelines
➤ Toy Regression (Sinusoid)

KAN:

python -m src.train.train_toy --activation kan --save_dir ./results/toy_kan


ReLU MLP:

python -m src.train.train_toy --activation relu --save_dir ./results/toy_relu

➤ CIFAR-10 Classification
Baseline CNN
python -m src.train.train_cifar --activation relu --save_dir ./results/cifar_relu

CNN + KAN Head
python -m src.train.train_cifar --activation kan --save_dir ./results/cifar_kan

CNN + Residual KAN Head
python -m src.train.train_cifar_residual --head residual_kan --save_dir ./results/cifar_residual/residual_kan

➤ IMDB Sentiment Classification
python -m src.nlp.train_imdb --save_dir ./results/nlp_imdb

➤ Tabular Regression (Housing & Energy)
python -m src.train.train_tabular --dataset housing
python -m src.train.train_tabular --dataset energy

📊 Analysis Pipelines
➤ Spline Visualization
python -m src.analysis.plot_splines \
  --model results/cifar_kan/cifar_model.pth \
  --model-type cnn \
  --save-dir results/interpretability/splines

➤ Knot Sensitivity
python -m src.analysis.knot_sensitivity \
  --model results/toy_kan/toy_model.pth \
  --dataset toy

➤ Locality and Support Width
python -m src.analysis.locality \
  --model results/cifar_kan/cifar_model.pth \
  --dataset cifar

➤ Activation Patterns
python -m src.analysis.activation_response \
  --kan-model results/cifar_kan/cifar_model.pth \
  --baseline-model results/cifar_relu/cifar_model.pth

🛡 Robustness Experiments
➤ Noise Robustness
python -m src.robustness.noise_robustness \
  --toy-kan results/toy_kan/toy_model.pth \
  --cifar-kan results/cifar_kan/cifar_model.pth

➤ Low Data Robustness
python -m src.robustness.low_data \
  --cifar-kan results/cifar_kan/cifar_model.pth

⚡ Efficiency Evaluation

Compute:

params

MACs

forward-time latency

model size

python -m src.analysis.compute_efficiency
python -m src.analysis.add_residual_to_efficiency

📈 Key Findings
✔ KAN consistently outperforms MLP and CNN on regression
✔ Residual KAN beats all models on CIFAR-10 (especially noise & low-data)
✔ Splines are interpretable: smooth derivatives, distinct knot importance
✔ KAN activations show strong locality → better interpretability
✔ Efficiency close to CNN despite higher flexibility
🧩 Interpretability Gallery (Available in /results)

Spline functions

Spline derivatives

Knot gradient importance

Activation heatmaps

Locality histograms

Support width histograms

Noise curves

Low-data curves

Efficiency bar charts

All included inside results/interpretability.
