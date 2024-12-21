# Fine-tuned-GPT-2-MedicalData
Medical Question Answering using Fine-tuned GPT-2


s fine-tuning a GPT-2 language model for medical question answering using the MedQuAD dataset.  The goal is to create a model capable of generating relevant and informative responses to medical queries.

## Project Details

### Dataset

The project utilizes the MedQuAD (Medical Question Answering Dataset) ([https://github.com/abachaa/MedQuAD/tree/master](https://github.com/abachaa/MedQuAD/tree/master)).  This dataset contains medical question-answer pairs, along with metadata such as question focus, UMLS (Unified Medical Language System) CUI (Concept Unique Identifier), semantic type, and semantic group.  More details about the dataset's construction can be found in this paper: [https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4).

The dataset is in CSV format with the following columns:

- **Focus**: The question's focus.
- **CUI**: Concept Unique Identifier.
- **SemanticType**: Semantic type of the medical concept.
- **SemanticGroup**: Semantic group of the medical concept.
- **Question**: The medical question.
- **Answer**: The corresponding answer.


### Problem Statement

The primary objective is to fine-tune a GPT-2 model on the MedQuAD dataset to enable accurate and relevant response generation for medical queries.  The existing, large medical literature necessitates a quick and effective way to retrieve answers, and a question-answering system fulfills this need better than traditional search engines.

### GPT-2 Model

This project employs the GPT-2 language model, known for its ability to generate coherent and contextually appropriate text.  

More details about GPT-2 can be found here: [http://jalammar.github.io/illustrated-gpt2/](http://jalammar.github.io/illustrated-gpt2/). 

**The model is fine-tuned to adapt its knowledge to the medical domain.**

## Technical Details

### Dependencies and Setup

The project relies on several Python libraries, including:

- transformers
- datasets
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- torch


The code begins with installing necessary libraries and handling potential version conflicts.

### Data Preprocessing and EDA

- Missing values are handled by filling with the mode of each column.
- Duplicate entries based on the `Question` and `Answer` columns are removed.
- Exploratory Data Analysis (EDA) is performed, specifically examining the distribution of questions across different `Focus` categories using bar plots of the top 100 categories.


### Data Preparation

- A training and validation set are created.  The training set contains 4 samples per `Focus` category for the top 100 categories (400 samples total). The validation set has 1 sample per `Focus` category from the remaining data (100 samples total).
- The `Question` and `Answer` text are combined into a single sequence using special tokens: `<question>`, `<answer>`, and `<end>`.
- These sequences are combined into training and validation text files.


### Model Fine-tuning

- A pre-trained GPT-2 tokenizer and model are loaded.
- Special tokens are added to the tokenizer.
- The training and validation data is tokenized.
- A `DataCollatorForLanguageModeling` object is used to handle padding and batching.
- The GPT-2 model is fine-tuned for 100 epochs, using the AdamW optimizer, gradient checkpointing, and a specified learning rate.
- The fine-tuned model and tokenizer are saved to a specified directory.


### Model Evaluation

- The fine-tuned model is tested with sample prompts.
- The performance of the fine-tuned model is compared against an untuned GPT-2 model.  Its observed that Finetuned GPT2 model performing far better than original GPT2 model.  

### Usage Instructions
* Ensure the correct environment, packages and data file (MedQuAD.csv) are available.
* Run the notebook cells in order, noting the data loading and model saving steps.
* After execution, you will have the saved fine-tuned model which can be used for generating medical responses(future purpose).
