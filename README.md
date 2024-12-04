This project implements a complete Machine Learning pipeline for training a Convolutional Neural Network for MNIST digit classification with automated CI/CD pipeline integration. What makes this project special is its focus on automated testing and continuous integration. 

At its heart, we have a Convolutional Neural Network implemented in PyTorch that's designed to recognize handwritten digits. The model architecture is deliberately kept simple yet effective, with a constraint of having fewer than 25,000 parameters to ensure efficiency and to get a training accuracy of more than 95% in 1 epoch. 

One of the key features of our pipeline is its comprehensive testing suite. When we run 'pytest', you'll notice several critical tests are being executed. These tests verify not just the basic functionality, but also ensure our model meets specific quality criteria - including a parameter count check and, most importantly, an accuracy threshold of 95%

One of the most powerful aspects of this pipeline is its integration with GitHub Actions. Every time we push our code to GitHub, an automated workflow springs into action. It creates a fresh environment, runs all our tests, trains the model, and saves it as an artifact. This automation ensures that our results are reproducible and that every code change meets our quality standards. In our .github/workflows/model-training.yml file, we've defined an automated workflow that triggers whenever code is pushed to the main branch or when a pull request is created. What makes this setup particularly valuable is its ability to catch issues early. If someone tries to push code that breaks our model's performance or exceeds our parameter budget, the workflow will fail, preventing problematic code from entering our main branch. You can see all these runs in the Actions tab of our GitHub repository, complete with logs and artifacts from each run.




