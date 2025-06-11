# 🧠 Custom AutoML System

A fully functional Custom AutoML system built to automate the entire machine learning pipeline from data preprocessing to model deployment. This system handles data preprocessing, model training, hyperparameter tuning, evaluation, and prediction through a modular and reusable architecture.

## 🚀 Features

- **Automated Preprocessing**: Smart handling of missing values, feature scaling, and categorical encoding
- **Multi-Model Training**: Trains multiple ML algorithms in parallel for optimal performance
- **Intelligent Hyperparameter Tuning**: Automated optimization for top-performing models
- **Comprehensive Evaluation**: Model selection based on key performance metrics
- **Easy Prediction Pipeline**: Seamless inference on new unseen data
- **Model Persistence**: Saves best model and preprocessing pipeline for future use
- **Modular Design**: Clean, extensible codebase for easy customization

## 📁 Project Structure

```
Custom-automl/
│
├── automl/                    # Core AutoML logic
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── model_training.py      # Multi-model training pipeline
│   ├── model_tuning.py        # Hyperparameter optimization
│   └── evaluator.py           # Model evaluation and selection
│
├── data/                      # Input datasets
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test/prediction dataset
│
├── outputs/                   # Generated outputs (auto-created)
│   ├── best_model.pkl        # Trained best model
│   ├── preprocessor.pkl      # Fitted preprocessing pipeline
│   └── predictions.csv       # Model predictions
│
├── main.py                   # Training pipeline entry point
├── predict.py                # Prediction script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download this repository
2. Navigate to the project directory
3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Usage

### 1. Training the AutoML Pipeline

Place your training dataset at `data/train.csv` with your target column included. Then run:

```bash
python main.py
```

**What happens during training:**
- Automatically detects data types and handles preprocessing
- Trains multiple ML algorithms (Random Forest, XGBoost, SVM, etc.)
- Performs hyperparameter tuning on top performers
- Evaluates models using cross-validation
- Selects and saves the best performing model
- Saves preprocessing pipeline for consistent transformations

### 2. Making Predictions

Place your new dataset at `data/test.csv` (same structure as training data, excluding target column). Then run:

```bash
python predict.py --input data/test.csv --target [TARGET_COLUMN_NAME]
```

**Example:**
```bash
python predict.py --input data/test.csv --target Survived
```

**What happens during prediction:**
- Loads the saved best model and preprocessor
- Applies identical transformations to new data
- Generates predictions and confidence scores
- Saves results to `outputs/predictions.csv`

## 📊 Supported Models

The system currently supports the following algorithms:
- Random Forest Classifier/Regressor
- XGBoost
- Support Vector Machine (SVM)
- Logistic Regression
- Gradient Boosting
- Extra Trees

*Note: The system automatically detects whether your problem is classification or regression.*

## 📈 Use Cases

- **Rapid Prototyping**: Quickly test ML feasibility on new datasets
- **Baseline Models**: Generate strong baselines for comparison
- **Non-Expert Friendly**: Enable domain experts to build ML models without deep technical knowledge
- **Research Tool**: Framework for AutoML research and experimentation
- **Production Ready**: Clean, modular code suitable for production environments

## 🎯 Configuration

### Customizing Models
Edit `automl/model_training.py` to add or remove algorithms from the training pipeline.

### Adjusting Hyperparameters
Modify the parameter grids in `automl/model_tuning.py` to customize search spaces.

### Preprocessing Options
Configure preprocessing steps in `automl/preprocessing.py` for domain-specific requirements.

## 📋 Requirements

```
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Performance Tips

- **Data Quality**: Clean data leads to better models - consider domain-specific preprocessing
- **Feature Engineering**: The system handles basic preprocessing, but domain features can improve results
- **Compute Resources**: Hyperparameter tuning can be time-intensive on large datasets
- **Memory Usage**: Large datasets may require chunking or sampling strategies

## 🛠️ Future Roadmap

### Planned Features
- [ ] **Model Explainability**: Integration with SHAP and LIME for interpretability
- [ ] **Advanced Feature Engineering**: Automated feature creation and selection
- [ ] **Model Monitoring**: Performance tracking and drift detection
- [ ] **Web Interface**: GUI for non-technical users
- [ ] **Cloud Deployment**: One-click deployment to AWS/GCP/Azure
- [ ] **Time Series Support**: Specialized handling for temporal data
- [ ] **Deep Learning**: Integration with neural network architectures

### Architecture Improvements
- [ ] **Distributed Training**: Multi-GPU and multi-node support
- [ ] **Pipeline Versioning**: Track and manage model versions
- [ ] **A/B Testing Framework**: Built-in experimentation capabilities
- [ ] **Real-time Inference**: Low-latency prediction API

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility where possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) section for existing solutions
2. Create a new issue with detailed information about your problem
3. Include sample data and error messages when possible

## 🎉 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/)
- Inspired by automated machine learning research and best practices
- Community feedback and contributions

---

**Ready to automate your machine learning workflow? Get started with Custom AutoML today!** 🚀

---
**Live-Link**
- [Preview](https://custom-automl.streamlit.app/)
