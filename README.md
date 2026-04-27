# 30 Day Data Science Path

A 30-day hands-on learning path for data science, machine learning, NLP, LLMs, OCR, document intelligence, and RAG.

The project is organized as Jupyter notebooks. Each notebook is one learning day with explanations, runnable code, tricky examples, exercises, and solutions.

## What You Will Learn

- Python, NumPy, pandas, data cleaning, and file I/O
- PyTorch basics and model training foundations
- Classical machine learning with scikit-learn and XGBoost
- Evaluation metrics, SHAP, and small ML projects
- Text preprocessing, TF-IDF, embeddings, transformers, and BERT fine-tuning
- OpenAI API usage, prompt engineering, structured output, and information extraction
- Local LLM workflows with Ollama
- OCR with Tesseract, EasyOCR, OpenCV, and LLM-assisted document pipelines
- Embeddings, chunking, vector databases, LangChain, ChromaDB, FAISS, and RAG chatbots

## Project Structure

```text
.
+-- Phase_1/
|   +-- Day_01_Phase1_PandasEssential.ipynb
|   `-- Day_05_Phase1_PyTorchBasics.ipynb
+-- Phase_2/
+-- Phase_3/
+-- Phase_4/
+-- Phase_5/
+-- Phase_6/
+-- README.md
`-- requirements.txt
```

## Phase Overview

| Phase | Days | Topic |
|---|---:|---|
| Phase 1 | 1-5 | Python data stack foundations |
| Phase 2 | 6-10 | Classical machine learning |
| Phase 3 | 11-15 | NLP and transformers |
| Phase 4 | 16-20 | LLM APIs, local models, and extraction |
| Phase 5 | 21-25 | OCR and document intelligence |
| Phase 6 | 26-30 | Embeddings, vector search, LangChain, and RAG |

## Notebook Map

| Day | File | Main topic |
|---:|---|---|
| 1 | `Phase_1/Day_01_Phase1_PandasEssential.ipynb` | pandas essentials |
| 2 | `Phase_1/Day_02_Phase1_NumpyVectorization.ipynb` | NumPy arrays and vectorization |
| 3 | `Phase_1/Day_03_Phase1_DataCleaningAndIO.ipynb` | data cleaning and file I/O |
| 4 | `Phase_1/Day_04_Phase1_FakerAndPythonCore.ipynb` | Faker, synthetic data, and Python core skills |
| 5 | `Phase_1/Day_05_Phase1_PyTorchBasics.ipynb` | PyTorch tensors, gradients, and neural network basics |
| 6 | `Phase_2/Day_06_Phase2_TrainTestAndPipelines.ipynb` | train/test split and ML pipelines |
| 7 | `Phase_2/Day_07_Phase2_LogRegAndTrees.ipynb` | logistic regression and decision trees |
| 8 | `Phase_2/Day_08_Phase2_EnsemblesAndXGBoost.ipynb` | ensemble models and XGBoost |
| 9 | `Phase_2/Day_09_Phase2_ClassificationMetrics.ipynb` | classification metrics |
| 10 | `Phase_2/Day_10_Phase2_ShapAndProject.ipynb` | SHAP and an ML project |
| 11 | `Phase_3/Day_11_Phase3_TextPreprocessingAndTFIDF.ipynb` | text preprocessing and TF-IDF |
| 12 | `Phase_3/Day_12_Phase3_EmbeddingsAndTransformers.ipynb` | embeddings and transformer concepts |
| 13 | `Phase_3/Day_13_Phase3_HuggingFaceBasics.ipynb` | Hugging Face pipelines, tokenizers, and models |
| 14 | `Phase_3/Day_14_Phase3_FineTuningBERT.ipynb` | fine-tuning BERT with Hugging Face Trainer |
| 15 | `Phase_3/Day_15_Phase3_ComplaintClassificationProject.ipynb` | complaint classification project |
| 16 | `Phase_4/Day_16_Phase4_OpenAIAPI.ipynb` | OpenAI API basics |
| 17 | `Phase_4/Day_17_Phase4_OllamaLocalModels.ipynb` | local LLMs with Ollama |
| 18 | `Phase_4/Day_18_Phase4_PromptEngineering.ipynb` | prompt engineering |
| 19 | `Phase_4/Day_19_Phase4_StructuredOutputExtraction.ipynb` | structured output extraction |
| 20 | `Phase_4/Day_20_Phase4_InformationExtractionProject.ipynb` | information extraction project |
| 21 | `Phase_5/Day_21_Phase5_TesseractBasics.ipynb` | Tesseract OCR basics |
| 22 | `Phase_5/Day_22_Phase5_EasyOCRAndComparison.ipynb` | EasyOCR and Tesseract comparison |
| 23 | `Phase_5/Day_23_Phase5_OpenCVPreprocessing.ipynb` | OpenCV preprocessing for OCR |
| 24 | `Phase_5/Day_24_Phase5_OcrLlmPipeline.ipynb` | OCR plus LLM pipeline |
| 25 | `Phase_5/Day_25_Phase5_DocumentIntelligenceProject.ipynb` | document intelligence project |
| 26 | `Phase_6/Day_26_Phase6_EmbeddingsDeepDive.ipynb` | embeddings deep dive |
| 27 | `Phase_6/Day_27_Phase6_ChunkingStrategies.ipynb` | chunking strategies |
| 28 | `Phase_6/Day_28_Phase6_ChromaAndFAISS.ipynb` | ChromaDB and FAISS |
| 29 | `Phase_6/Day_29_Phase6_LangChainRetrievalQA.ipynb` | LangChain retrieval QA |
| 30 | `Phase_6/Day_30_Phase6_RagChatbotProject.ipynb` | RAG chatbot project |

## Setup

Use Python 3.10 or newer. Python 3.11 is recommended.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in day order.

## Optional External Tools

Some notebooks work with mock fallbacks, but these tools unlock the full examples.

### OpenAI API

Used in the OpenAI, structured extraction, embeddings, and RAG notebooks.

Set your API key before running API cells:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Ollama

Used for local LLM examples.

Install Ollama from the official site, then pull a model:

```bash
ollama pull mistral
ollama pull llama3.2
```

Start the local server if needed:

```bash
ollama serve
```

The notebooks expect Ollama at:

```text
http://localhost:11434
```

### Tesseract OCR

Used in OCR notebooks.

On macOS:

```bash
brew install tesseract
brew install tesseract-lang
```

On Ubuntu or Debian:

```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-deu tesseract-ocr-tur
```

The Python wrapper is included in `requirements.txt` as `pytesseract`, but the system Tesseract binary must be installed separately.

## Recommended Workflow

1. Activate the virtual environment.
2. Open one notebook at a time.
3. Run cells from top to bottom.
4. Complete exercises before opening solution sections.
5. Keep notes on errors, model behavior, and concepts that need review.
6. Move to the next day only after the current notebook runs cleanly.

## Notes About Downloads

Some notebooks may download models or data the first time they run:

- Hugging Face models can take time and disk space.
- `sentence-transformers` downloads embedding models.
- EasyOCR may download OCR model weights.
- ChromaDB and FAISS examples may create local in-memory indexes.

## Troubleshooting

### Jupyter cannot find installed packages

Make sure Jupyter is running from the same virtual environment:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name data-science-path
jupyter notebook
```

Then select the `data-science-path` kernel inside Jupyter.

### Tesseract is not found

Install the system binary, then restart the terminal and Jupyter.

Check it with:

```bash
tesseract --version
```

### OpenAI cells use mock responses

Check that `OPENAI_API_KEY` is set in the same terminal where Jupyter was started.

### Ollama connection fails

Check that Ollama is installed, running, and has a model pulled:

```bash
ollama list
ollama serve
```

### Hugging Face model download fails

Check your internet connection. If needed, rerun the cell after the download finishes or clears.

## Main Python Libraries

- Data work: `numpy`, `pandas`, `openpyxl`, `faker`
- Visualization: `matplotlib`
- Machine learning: `scikit-learn`, `xgboost`, `shap`
- Deep learning: `torch`
- NLP: `nltk`, `transformers`, `datasets`, `sentence-transformers`
- LLM APIs: `openai`, `requests`, `pydantic`
- OCR and images: `pillow`, `pytesseract`, `opencv-python`, `easyocr`
- RAG and vector search: `chromadb`, `faiss-cpu`, `langchain`, `langchain-community`, `langchain-openai`

## Suggested Order

Run the notebooks in numerical order from Day 1 to Day 30. Later notebooks assume concepts from earlier phases.

If you already know the basics:

- Start at Day 6 for classical ML.
- Start at Day 11 for NLP.
- Start at Day 16 for LLM APIs.
- Start at Day 21 for OCR.
- Start at Day 26 for RAG.
