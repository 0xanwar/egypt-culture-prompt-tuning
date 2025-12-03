# Egyptian Cultural Understanding with Prompt Tuning

![Egyptian Cultural AI](https://img.shields.io/badge/Egyptian_Culture-AI-blue?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

This repository implements a **parameter-efficient approach** to Egyptian cultural understanding using **prompt tuning** on BERT. The system classifies social intents in Egyptian cultural contexts into 5 core themes, addressing bias in vision-language models through culturally-aware text modeling.

## 🌟 Features

- **Prompt Tuning Architecture**: Adapts BERT with only 0.01% trainable parameters
- **Egyptian Cultural Themes**: Recognizes 5 core cultural dimensions:
  - Religious Celebration (Eid, Mawlid, Ramadan)
  - Family and Respect (elder care, familial bonds)
  - National Pride (Revolution Day, patriotism)
  - Cultural Heritage (Sham El Nessim, traditional foods)
  - Community Generosity (neighborhood gift-giving)
- **Parameter Efficiency**: Updates only 11,520 parameters out of 110M
- **Bias Mitigation**: Counters Western defaults in pretrained language models
- **Modular Design**: Clean OOP structure for easy extension

## 📁 Project Structure

```
egypt-culture-prompt-tuning/
├── src/                  # Core source code
│   ├── config.py         # Configuration classes
│   ├── data/             # Dataset handling
│   ├── models/           # Model architectures
│   ├── training/         # Training and evaluation logic
│   └── utils/            # Utility functions
├── scripts/              # Executable scripts
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── notebooks/            # Experiment notebooks
├── models/               # Trained models
├── data/                 # Raw dataset files
└── configs/              # Configuration files
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/egyptian_cultural_ai.git
cd egyptian_cultural_ai

# Create virtual environment (recommended)
python -m venv env
source env/bin/activate  # Linux/MacOS
# env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train the model on Anwar12/Atomic-EgMM dataset
python scripts/train.py
```

### Evaluation

```bash
# Evaluate trained model with top-2 predictions
python scripts/evaluate.py
```

## 📊 Sample Output

```
🔍 TOP-2 CULTURAL THEME PREDICTIONS
================================================================================

Text: PersonX visits elderly relatives with gifts after Eid prayer....
--------------------------------------------------------------------------------
  1. family_and_respect           (ID: 1) | Confidence: 0.856
  2. religious_celebration        (ID: 0) | Confidence: 0.065

Text: PersonX prepares salted fish and colored eggs for family during Sham...
--------------------------------------------------------------------------------
  1. cultural_heritage            (ID: 3) | Confidence: 0.424
  2. family_and_respect           (ID: 1) | Confidence: 0.300
  💡 Cultural ambiguity detected!
```

## 🧠 Technical Details

### Architecture
- **Base Model**: BERT-base-uncased (frozen)
- **Prompt Tokens**: 10 learnable embeddings inserted after [CLS] token
- **Classification Head**: Linear layer mapping to 5 cultural themes
- **Total Trainable Parameters**: 11,520 (0.01% of BERT)

### Dataset
- **Source**: [Atomic-EgMM](https://huggingface.co/datasets/Anwar12/Atomic-EgMM)
- **Size**: 125 training / 36 validation / 18 test examples
- **Cultural Mapping**: Rule-based intent-to-theme mapping using Egyptian keywords

### Training Configuration
- **Epochs**: 15
- **Batch Size**: 8 (adaptive for small datasets)
- **Learning Rates**: 
  - Prompt embeddings: 5e-4
  - Classifier head: 1e-5

## 🎯 Use Cases

- **Cultural Bias Auditing**: Identify Western defaults in VLMs
- **Egyptian Content Moderation**: Understand culturally appropriate content
- **Social Media Analysis**: Analyze Egyptian cultural events in text
- **Vision-Language Extension**: Provide cultural context for image understanding

## 📈 Results

| Cultural Theme | Confidence | Status |
|----------------|------------|--------|
| Family & Religious | >75% | ✅ Strong |
| Cultural Heritage | ~42% | ⚠️ Needs more data |
| National Pride | 0% | ❌ Critical gap |
| Community Generosity | ~36% | ⚠️ Ambiguous |

> **Recommendation**: Add 10-15 examples per underrepresented theme for comprehensive coverage

## 🤝 Contributing

Contributions to improve Egyptian cultural representation are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Dataset: [Anwar12/Atomic-EgMM](https://huggingface.co/datasets/Anwar12/Atomic-EgMM)
- Base Model: [BERT](https://huggingface.co/bert-base-uncased)
- Inspiration: [Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)
- Cultural Expertise: Egyptian cultural consultants and linguists