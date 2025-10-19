# Nuclear Calibration with Machine Learning

## 📊 Project Overview
A machine learning system for calibrating nuclear spectra using linear regression. This project demonstrates the application of ML in scientific data analysis with proper validation and outlier detection.

## 🎯 Key Features
- Linear regression for energy-channel calibration
- Automatic detection and exclusion of problematic data (⁷Li peaks)
- Professional visualization with particle-specific colors
- Quantitative error analysis (MAE, MAPE, RMSE)
- Comparison with ideal physical model

## 🚀 Results
- **Calibration Equation**: Energy = 0.0037 × Channel + 0.3553
- **Prediction Error for ⁷Li**: 1.93 MeV (MAE), 13.85% (MAPE)
- **Validation**: Confirmed hypothesis that ⁷Li peaks introduce significant calibration errors

## 🛠️ Installation
```bash
git clone https://github.com/tu-usuario/nuclear-calibration-ml.git
cd nuclear-calibration-ml
pip install -r requirements.txt

💻 Usage
Jupyter Notebook:

from src.calibration import NuclearCalibration

calibration = NuclearCalibration()
calibration.train_models()
calibration.plot_separate_graphs(save_figures=True)

Python Script:
python src/calibration.py

📁 Project Structure
nuclear_calibration/
├── notebooks/          # Jupyter notebook for analysis
├── src/               # Source code
├── figures/           # Generated plots
├── requirements.txt   # Dependencies
└── README.md         # This file

🔬 Scientific Context
This project addresses a common problem in nuclear spectroscopy: calibrating detector channels to energy values. The ML approach provides a quantitative method for identifying and handling problematic measurements.

🎓 Skills Demonstrated
Machine Learning (scikit-learn)

Data Analysis & Visualization

Scientific Computing

Statistical Validation

Python Programming


