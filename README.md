# **Transformers Implementation**

This repository contains a modular and flexible implementation of the **Transformer** architecture and its variants. The repository incorporates key advancements in Transformer-based models for both natural language processing (NLP) and computer vision (CV) tasks.

## **Key Features**

- **Transformer Implementation**: Based on the foundational work, *Attention is All You Need* by Vaswani et al. (2017).
- **Gaussian Error Linear Unit (GELU)**: Uses the GELU activation function, introduced in *Improving Language Understanding by Generative Pre-Training* (Radford, 2018).
- **Pre-Layer Normalization**: Includes support for pre-layer normalization following Xiong et al. (2020).
- **Gated Linear Units (GLUs)**: Implements GLU and its variants as described by Dauphin et al. (2017) and Shazeer (2020).
- **Learnable Positional Encoding**: Supports learnable positional embeddings, inspired by BERT (Devlin et al., 2019).
- **Visualization Tools**: Includes methods to visualize attention flow, based on Abnar & Zuidema (2020).
- **Vision Transformer (ViT)**: Extends Transformer models for vision tasks, based on Dosovitskiy et al. (2020).

---

## **Reference Works**

### **Transformer**
- Vaswani, A. *Attention is all you need.* Advances in Neural Information Processing Systems (2017).

### **Key Innovations**
- **Gaussian Error Linear Unit (GELU)**:  
  Radford, Alec. *Improving language understanding by generative pre-training.* (2018).  
- **Pre-Layer Normalization**:  
  Xiong, Ruibin, et al. *On layer normalization in the transformer architecture.* International Conference on Machine Learning (2020).  
- **Gated Linear Unit (GLU)**:  
  Dauphin, Yann N., et al. *Language modeling with gated convolutional networks.* International Conference on Machine Learning (2017).  
  Shazeer, Noam. *Glu variants improve transformer.* arXiv preprint arXiv:2002.05202 (2020).  
- **Learnable Positional Encoding**:  
  Devlin, Jacob, et al. *BERT: Pre-training of deep bidirectional transformers for language understanding.* Proceedings of NAACL-HLT (2019).  
- **Visualization**:  
  Abnar, Samira, and Willem Zuidema. *Quantifying attention flow in transformers.* arXiv preprint arXiv:2005.00928 (2020).  

### **Vision Transformer**
- Dosovitskiy, Alexey. *An image is worth 16x16 words: Transformers for image recognition at scale.* arXiv preprint arXiv:2010.11929 (2020).

---

## **Acknowledgements**
This repository draws inspiration from key research works in Transformer architectures and their innovations. For detailed explanations and mathematical formulations, please refer to the cited papers.

## **License**
This project is licensed under the MIT License.
