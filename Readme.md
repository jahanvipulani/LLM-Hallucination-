# Comparative Evaluation of Lexical Overlap and Natural Language Inference for LLM Hallucination Detection

## 1. About the Project

Large Language Models (LLMs) can generate responses that contain incorrect, unsupported, or fabricated information. These errors are commonly referred to as **hallucinations**.

This research project evaluates two approaches for identifying hallucinations in LLM-generated responses:

1. **Lexical Overlap** — estimates how much of a generated response is supported by the available source text using word overlap.
2. **Natural Language Inference (NLI)** — uses a pretrained language model to estimate whether a response is entailed by, contradicts, or is neutral with respect to the source text.

A majority-class baseline is also included to provide a simple reference point for evaluating the two detection methods.

The experiments use the **RAGTruth dataset** and compare the methods using standard classification metrics.

### Research Question

How accurately and reliably do lexical overlap and natural language inference identify responses containing annotated hallucinations?

## 2. Research Objectives

- Evaluate lexical overlap as a baseline hallucination detection method.
- Evaluate an NLI-based method using a pretrained transformer model.
- Compare the methods using accuracy, precision, recall, F1-score, and AUROC.
- Study how classification thresholds affect detection performance.
- Examine the limitations of automated hallucination detection.

## 3. Methods Used

### A. Majority-Class Baseline

The baseline predicts the majority class for every test response. In this experiment, it predicts that every response contains no annotated hallucination.

This provides a reference point and demonstrates why accuracy alone is not sufficient for evaluating hallucination detection.

### B. Lexical Overlap

The lexical overlap method compares words in the generated response with words in the associated source text.

The source text is formed from the available `source` and `source_info` fields. Text is converted to lowercase, tokenized using whitespace, and stripped of surrounding punctuation.

The overlap score is calculated as:

\[
\text{Overlap}(R,S)=
\frac{|\{t\in R:t\in S\}|}{|R|}
\]

The unsupported score is:

\[
\text{Unsupported Score}=1-\text{Overlap}(R,S)
\]

Where:
- \(R\) represents the response tokens.
- \(S\) represents the set of source tokens.

A response is predicted to contain a hallucination when its unsupported score meets or exceeds the selected threshold.

**Limitation:** Different wording, paraphrasing, and valid inferences may receive low lexical overlap scores. High word overlap also does not guarantee factual correctness.

### C. Natural Language Inference (NLI)

The NLI method uses the pretrained model:

`cross-encoder/nli-deberta-v3-small`

The source text is supplied as the premise, and the generated response is supplied as the hypothesis.

The model estimates three relationships:

- **Entailment:** The source supports the response.
- **Contradiction:** The response conflicts with the source.
- **Neutral:** The source neither establishes nor contradicts the response.

The unsupported score is calculated as:

\[
\text{Unsupported Score} =
P(\text{Contradiction})+P(\text{Neutral})
\]

A response is predicted to contain a hallucination when this score meets or exceeds the selected threshold.

**Limitation:** A neutral prediction does not necessarily mean the response is false. The model can also miss important context when input text exceeds its 512-token maximum sequence length.

## 4. Dataset

This project uses the **RAGTruth** dataset, which contains retrieval-augmented generation responses annotated for hallucinations.

Dataset source: [RAGTruth — ACL Anthology](https://aclanthology.org/2024.acl-long.585/)

The project uses two JSON Lines files:

- `source_info.jsonl` — contains source information, prompts, and source identifiers.
- `response.jsonl` — contains generated responses, split information, and hallucination annotations.

The files are joined using `source_id`.

### Dataset Summary

| Item | Count |
|---|---:|
| Source records | 2,965 |
| Total response records | 17,790 |
| Training responses | 15,090 |
| Test responses | 2,700 |
| Test responses with annotated hallucination spans | 943 |
| Test responses without annotated hallucination spans | 1,757 |

**Important:** A response with a non-empty annotation is treated as positive. A response without an annotated hallucination span is treated as negative for this experiment; this does not guarantee that the response is completely factual.

## 5. Project Structure

The main files are organized as follows:

```text
LLM Hallucination research paper/
│
├── data/
│   ├── source_info.jsonl
│   └── response.jsonl
│
├── src/
│   ├── inspect_datasets.py
│   ├── dataset_summary.py
│   ├── inspect_ragtruth.py
│   ├── check_labels.py
│   ├── check_general.py
│   ├── baseline.py
│   ├── baseline1.py
│   ├── evaluate_overlap.py
│   ├── evaluate_nli.py
│   ├── tune_overlap_validation.py
│   ├── tune_nli_validation.py
│   ├── compare_methods.py
│   └── verify_experiment.py
│
├── README.md
└── .venv313/
```

*Note: This is an overview of the main scripts. Exact filenames may vary slightly depending on the current repository version. The virtual environment folder should not be committed to GitHub.*

## 6. Requirements

- Python 3.13, matching the environment used for the experiments
- Git (optional, for cloning the repository)
- VS Code or another Python IDE
- Internet access for downloading the pretrained NLI model the first time it is needed
- The RAGTruth dataset files
- Python packages required by the individual scripts, including NumPy, scikit-learn, and Transformers, as applicable

The scripts use Python's built-in `csv`, `json`, and other standard-library modules for file processing, avoiding a dependency on pandas in the current Windows environment.

## 7. Getting Started

### Step 1: Clone the Repository

Replace the placeholder with the actual URL of this repository.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "LLM Hallucination research paper"
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 2: Open the Project

Open the project folder in VS Code.

In the VS Code terminal, navigate to the project root directory:

```powershell
cd "path\to\LLM Hallucination research paper"
```

### Step 3: Create a Virtual Environment

If you are creating a fresh environment, run:

```powershell
py -3.13 -m venv .venv313
```

Activate it in PowerShell:

```powershell
.\.venv313\Scripts\Activate.ps1
```

If you already have a working `.venv313` environment, activate it instead of recreating it.

### Step 4: Install Dependencies

Install the packages required by the scripts:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn transformers torch
```

If a `requirements.txt` file is provided in the repository, prefer installing the pinned dependencies with:

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Check the Dataset

Make sure the following files are present:

```text
data