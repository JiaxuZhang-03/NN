# NN

A collection of implementations and experiments with **Neural Networks** using PyTorch.  
This repository is designed for learning, experimenting, and building deep learning models from scratch, covering both foundational architectures and more advanced experiments.

---

## 📌 Features

- Implementations of classical neural network models:
  - Perceptron, MLP
  - CNNs (LeNet, AlexNet, VGG, ResNet, etc.)
  - RNNs / LSTMs / GRUs
- Training scripts with support for GPU / Apple Silicon (MPS)
- Modular structure for extending and experimenting with custom models
- Examples of:
  - Data preprocessing
  - Training loops with metrics (loss, accuracy)
  - Visualization of results (matplotlib)

---

## 🗂️ Repository Structure

```
NN/
│── data/                # Example datasets or dataset loaders
│── models/              # Implementations of neural network architectures
│   ├── mlp.py
│   ├── lenet.py
│   ├── vgg.py
│   └── resnet.py
│── utils/               # Helper functions (metrics, plotting, etc.)
│── train.py             # General training script
│── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/JiaxuZhang-03/NN.git
cd NN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run an experiment
```bash
python train.py --model lenet --dataset MNIST --epochs 5
```

---

## 🖼️ Example

Training a simple LeNet on MNIST:
```
Epoch [1/5] - Loss: 0.132 - Accuracy: 97.8%
Epoch [2/5] - Loss: 0.095 - Accuracy: 98.4%
...
```

---

## 📚 Roadmap

- [ ] Add Transformer-based models  
- [ ] Add more datasets (CIFAR-100, TinyImageNet)  
- [ ] Add logging with TensorBoard  
- [ ] Add pre-trained weights and evaluation scripts  

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome!  
Feel free to fork the repo and submit improvements.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
