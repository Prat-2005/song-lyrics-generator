# 🎤 Song Lyrics Generator

## Project Overview
This project uses a word-level LSTM trained on song lyrics to generate new text from a prompt. The model can create original song lyrics based on a short input phrase.

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Vocabulary Size | 4,981 |
| Embedding Dimension | 128 |
| Hidden Dimension | 512 |
| LSTM Layers | 2 |
| Dropout | 0.2 |
| Sequence Length | 20 |
| Total Parameters | ~ |

## Installation

```bash
git clone <your-repo-url>
cd lyrics-generator
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Example Outputs

**Prompt:** `love is`
> love is not just a feeling but a choice we make to stay together through all the storms

**Prompt:** `i am`
> i am the one who walks through shadows searching for a light that i can hold

**Prompt:** `you better`
> you better run fast or the night will catch you and pull you into its endless dreams

## Screenshots

> Add screenshots here after running the app locally.

## Project Structure

```
lyrics-generator/
├── models/
├── notebooks/
├── src/
│   ├── model.py
│   ├── inference.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```