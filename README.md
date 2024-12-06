# MNIST CNN Training Pipeline

![Model Accuracy](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/badges/accuracy.json)

This project implements a complete Machine Learning pipeline for training a Convolutional Neural Network (CNN) for MNIST digit classification with automated CI/CD pipeline integration. What makes this project special is its focus on automated testing and continuous integration. 

At its heart, we have a Convolutional Neural Network implemented in PyTorch that's designed to recognize handwritten digits. The model architecture is deliberately kept simple yet effective, with a constraint of having fewer than 25,000 parameters to ensure efficiency and to get a training accuracy of more than 95% in 1 epoch. 

One of the key features of our pipeline is its comprehensive testing suite. When we run 'pytest', you'll notice several critical tests are being executed. These tests verify not just the basic functionality, but also ensure our model meets specific quality criteria - including a parameter count check and, most importantly, an accuracy threshold of 95%

One of the most powerful aspects of this pipeline is its integration with GitHub Actions. Every time we push our code to GitHub, an automated workflow springs into action. It creates a fresh environment, runs all our tests, trains the model, and saves it as an artifact. This automation ensures that our results are reproducible and that every code change meets our quality standards. In our .github/workflows/model-training.yml file, we've defined an automated workflow that triggers whenever code is pushed to the main branch or when a pull request is created. What makes this setup particularly valuable is its ability to catch issues early. If someone tries to push code that breaks our model's performance or exceeds our parameter budget, the workflow will fail, preventing problematic code from entering our main branch. You can see all these runs in the Actions tab of our GitHub repository, complete with logs and artifacts from each run.

## Table of Contents
1. [Project Features](#project-features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Automated CI/CD Pipeline](#automated-cicd-pipeline)
5. [Testing](#testing)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Features

- **Convolutional Neural Network**: 
  - Implemented in PyTorch.
  - Designed to recognize handwritten digits from the MNIST dataset.
  - Optimized for efficiency, with fewer than 25,000 parameters.
  - Achieves a training accuracy of over 95% in just one epoch.

- **Comprehensive Testing Suite**: 
  - Includes critical tests using `pytest` to validate functionality, parameter constraints, and accuracy thresholds.

- **CI/CD Integration**: 
  - Automated workflows using GitHub Actions.
  - Reproducible results with automated environment setup, testing, training, and artifact generation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhavanagopakumar/MNIST_CICD_MLOps.git
   cd MNIST_CICD_MLOps
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Train the Model**:
   ```bash
   python train.py
   ```

2. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

---

## Automated CI/CD Pipeline

This project is integrated with GitHub Actions to ensure automated testing and training:

- **Workflow Triggers**: The pipeline is triggered on the following events:
  - Code pushed to the `main` branch.
  - Creation of pull requests.

- **Workflow Steps**:
  1. Creates a fresh virtual environment.
  2. Installs dependencies.
  3. Runs tests to validate model performance and parameter constraints.
  4. Trains the model and saves it as an artifact.

- **Defined Workflow File**: See `.github/workflows/model-training.yml` for details on the CI/CD configuration.

- **Failure Prevention**: The workflow halts if:
  - Model performance falls below the accuracy threshold.
  - Model exceeds the parameter count.

---

## Testing

The project includes a comprehensive testing suite to ensure reliability:

- **Critical Tests**:
  - Validate basic functionality.
  - Ensure model has fewer than 25,000 parameters.
  - Verify model achieves at least 95% training accuracy.

- Run tests locally using:
  ```bash
  pytest tests/ -v
  ```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Create a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out this project! If you have any questions or feedback, feel free to reach out.
