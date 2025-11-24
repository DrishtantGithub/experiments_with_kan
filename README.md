KAN-Enhanced Deep Learning Project

- Drishtant Jain (24M1085)
- Ankit Kumar Singh (24M1080)
- Nitin Tomar (24M1079)

This project implements and evaluates Kolmogorov–Arnold Networks (KANs) across multiple modalities and introduces a novel architecture, the Residual KAN Head. The work includes complete model implementations, training pipelines, interpretability analyses, robustness experiments, efficiency evaluations.

Summary

Developed a unified benchmarking framework for evaluating KANs against MLP and CNN baselines on vision (CIFAR-10), NLP (IMDB), tabular regression (Housing & Energy), and toy sinusoidal regression tasks.

Proposed a new Residual KAN Head architecture integrating a linear skip connection with a KAN spline block, improving stability, smoothness, and robustness with minimal parameter overhead.

Designed extensive experimental pipelines including knot ablations, spline curvature analysis, noise robustness, low-data generalization, and model efficiency comparisons (parameters, MACs, latency, model size).

Built an interpretability suite for spline visualization, derivative smoothness, knot sensitivity, activation mapping, locality estimation, and support-width analysis.

Implemented full reproducible pipelines with modular source code, experiment organization, and structured results storage.

Repository Structure (High-Level)

src/: All source code, including model definitions (MLP, CNN, KAN, Residual KAN Head), training modules, interpretability tools, robustness tests, and utilities.

results/: Trained models, evaluation metrics, plots, interpretability outputs, and robustness experiment results for all datasets.

paper/: IEEE-format LaTeX source files and project figures.

requirements.txt / environment.yml: Full environment and dependency specification.

Training Pipelines
Toy Regression (Sinusoid)
python -m src.train.train_toy --activation kan --save_dir ./results/toy_kan
python -m src.train.train_toy --activation relu --save_dir ./results/toy_relu

CIFAR-10 Classification
python -m src.train.train_cifar --activation relu --save_dir ./results/cifar_relu
python -m src.train.train_cifar --activation kan --save_dir ./results/cifar_kan
python -m src.train.train_cifar_residual --head residual_kan --save_dir ./results/cifar_residual

IMDB Sentiment Classification
python -m src.nlp.train_imdb --save_dir ./results/nlp_imdb

Tabular Regression (Housing, Energy)
python -m src.train.train_tabular --dataset housing
python -m src.train.train_tabular --dataset energy

Interpretability Tools
Spline Visualization
python -m src.analysis.plot_splines \
    --model results/cifar_kan/cifar_model.pth \
    --model-type cnn \
    --save-dir results/interpretability/splines

Knot Importance
python -m src.analysis.knot_sensitivity \
    --model results/toy_kan/toy_model.pth \
    --dataset toy

Locality and Support Width
python -m src.analysis.locality \
    --model results/cifar_kan/cifar_model.pth \
    --dataset cifar

Activation Pattern Comparison
python -m src.analysis.activation_response \
    --kan-model results/cifar_kan/cifar_model.pth \
    --baseline-model results/cifar_relu/cifar_model.pth

Robustness Experiments
Noise Robustness
python -m src.robustness.noise_robustness \
    --toy-kan results/toy_kan/toy_model.pth \
    --cifar-kan results/cifar_kan/cifar_model.pth

Low-Data Robustness
python -m src.robustness.low_data \
    --cifar-kan results/cifar_kan/cifar_model.pth

Efficiency Evaluation
python -m src.analysis.compute_efficiency
python -m src.analysis.add_residual_to_efficiency


Outputs include parameter counts, MACs, latency benchmarks, and model size.

Key Findings

KANs consistently outperform MLPs on all regression benchmarks.

KAN Head architectures outperform classical CNNs on CIFAR-10.

The proposed Residual KAN Head provides the best overall performance across accuracy, stability, and robustness.

KAN models demonstrate strong interpretability: smooth derivatives, meaningful spline behaviors, localized activations, and identifiable knot importance patterns.

KAN-based models exhibit higher resilience to noise and superior generalization in low-data settings.

Efficiency remains close to CNN baselines with small parameter overhead.

