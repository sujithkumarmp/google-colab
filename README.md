# Google Colab Machine Learning & AI Notebooks Collection

This repository contains a comprehensive collection of Jupyter notebooks demonstrating various machine learning, artificial intelligence, and mathematical computing concepts. All notebooks are designed to run on Google Colab with easy-to-follow implementations.

## 📚 Repository Structure

### Core Machine Learning Fundamentals
- **[simple.ipynb](simple.ipynb)** - Gradient Descent Visualization and Optimization
- **[simple2.ipynb](simple2.ipynb)** - Neural Network from Scratch using NumPy
- **[simple3.ipynb](simple3.ipynb)** - Symbolic Mathematics with SymPy
- **[word2vec.ipynb](word2vec.ipynb)** - Word2Vec Word Embeddings Implementation

### Advanced AI & NLP (`gen-ai/` directory)
- **[RNN.ipynb](gen-ai/RNN.ipynb)** - Recurrent Neural Networks with PyTorch
- **[cbow.ipynb](gen-ai/cbow.ipynb)** - Continuous Bag of Words (CBOW) Model
- **[word2vec.ipynb](gen-ai/word2vec.ipynb)** - Advanced Word2Vec Implementation
- **[ngram.ipynb](gen-ai/ngram.ipynb)** - N-gram Language Model
- **[gen_ai_from_scratch.ipynb](gen-ai/gen_ai_from_scratch.ipynb)** - Comprehensive Generative AI Tutorial
- **[Rnn_encode_deconder.ipynb](gen-ai/Rnn_encode_deconder.ipynb)** - RNN Encoder-Decoder Architecture
- **[1tokenizer.ipynb](gen-ai/1tokenizer.ipynb)** - Text Tokenization Fundamentals
- **[hello_tokeization.ipynb](gen-ai/hello_tokeization.ipynb)** - Basic Tokenization Introduction
- **[huggingFace_tokeization.ipynb](gen-ai/huggingFace_tokeization.ipynb)** - HuggingFace Tokenizers

## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended)
Each notebook includes a "Open in Colab" badge at the top. Simply click it to open and run the notebook in Google Colab with all dependencies pre-installed.

### Option 2: Local Setup
If you prefer to run locally, you'll need:

```bash
# Clone the repository
git clone https://github.com/sujithkumarmp/google-colab.git
cd google-colab

# Install required dependencies
pip install numpy matplotlib sympy torch torchvision transformers huggingface-hub
```

## 📖 Notebook Descriptions

### Machine Learning Fundamentals

#### [simple.ipynb](simple.ipynb) - Gradient Descent Visualization
- **Purpose**: Demonstrates gradient descent optimization algorithm
- **Key Concepts**: Loss functions, gradient computation, optimization paths
- **Libraries**: NumPy, Matplotlib, 3D plotting
- **Learning Outcomes**: Understanding how gradient descent finds optimal solutions

#### [simple2.ipynb](simple2.ipynb) - Neural Network from Scratch
- **Purpose**: Implements a complete neural network without ML frameworks
- **Key Concepts**: Forward propagation, backpropagation, parameter updates
- **Libraries**: NumPy
- **Learning Outcomes**: Deep understanding of neural network internals

#### [simple3.ipynb](simple3.ipynb) - Symbolic Mathematics
- **Purpose**: Explores symbolic mathematical computations
- **Key Concepts**: Differentiation, integration, symbolic expressions
- **Libraries**: SymPy
- **Learning Outcomes**: Mathematical foundations for machine learning

### Natural Language Processing & AI

#### [word2vec.ipynb](word2vec.ipynb) - Word Embeddings
- **Purpose**: Creates word vector representations
- **Key Concepts**: Skip-gram model, word embeddings, semantic similarity
- **Libraries**: PyTorch
- **Learning Outcomes**: Understanding word representations in NLP

#### [RNN.ipynb](gen-ai/RNN.ipynb) - Recurrent Neural Networks
- **Purpose**: Implements basic RNN for sequence modeling
- **Key Concepts**: Sequence processing, hidden states, temporal dependencies
- **Libraries**: PyTorch
- **Learning Outcomes**: Foundation for advanced sequence models

#### [cbow.ipynb](gen-ai/cbow.ipynb) - Continuous Bag of Words
- **Purpose**: Alternative approach to word embeddings
- **Key Concepts**: Context window, word prediction, embeddings
- **Libraries**: PyTorch
- **Learning Outcomes**: Different perspectives on word representation learning

## 🛠 Technologies Used

- **Python**: Primary programming language
- **NumPy**: Numerical computations and array operations
- **PyTorch**: Deep learning framework for neural networks
- **Matplotlib**: Data visualization and plotting
- **SymPy**: Symbolic mathematics
- **HuggingFace Transformers**: State-of-the-art NLP models
- **Google Colab**: Cloud-based Jupyter notebook environment

## 📋 Prerequisites

- Basic Python programming knowledge
- Understanding of linear algebra and calculus
- Familiarity with machine learning concepts (helpful but not required)
- Google account for Colab access

## 🎯 Learning Path

For beginners, we recommend following this order:

1. **Mathematical Foundation**: Start with `simple3.ipynb` for symbolic math
2. **Optimization Basics**: Move to `simple.ipynb` for gradient descent
3. **Neural Networks**: Progress to `simple2.ipynb` for neural network internals
4. **Word Embeddings**: Explore `word2vec.ipynb` for NLP foundations
5. **Advanced Topics**: Dive into the `gen-ai/` directory notebooks

## 🤝 Contributing

Feel free to contribute by:
- Adding new educational notebooks
- Improving existing implementations
- Fixing bugs or typos
- Adding documentation or comments

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⭐ Getting Started

1. Choose a notebook based on your learning goals
2. Click the "Open in Colab" badge
3. Follow the code and comments
4. Experiment with parameters and modifications
5. Build upon the concepts for your own projects

Happy learning! 🚀