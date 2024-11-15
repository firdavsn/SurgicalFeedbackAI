# Automating Feedback Analysis in Surgical Training: Detection, Categorization, and Assessment
---
#### Firdavs Nasriddinov*, Rafal Kocielnik*, Arushi Gupta, Cherine Yang, Elyssa Wong, Anima Anandkumar, Andrew J. Hung

This official repository holds code for the paper "**Automating Feedback Analysis in Surgical Training: Detection, Categorization, and Assessment**". Our [Paper](link) is accepted at [ML4H 2024](link). We open source all code and results here under a [permissive MIT license](LICENSE), to encourage reproduction and further research exploration. 

<img width="1366" alt="Framework" src="figures/main_figure.png">
<hr>

## Overview

This project aims to automate the analysis of feedback in surgical training, specifically focusing on detecting, categorizing, and assessing feedback provided during surgery training sessions using the da Vinci surgical system.

## Project Structure

```
.gitignore
code/
    analysis-analyze_dialogue_metrics.ipynb
    analysis-analyze_temporal_metrics.ipynb
    analysis-confusion_matrices.ipynb
    analysis-test_similarity_thresh.ipynb
    demo-dialogue_from_clip.ipynb
    exps-dialogue.py
    exps-temporal_detection_unseen_case.py
    exps-temporal_detection_unseen_surgeon.py
    misc-assign_anchors.ipynb
    misc-example_classify_dialogue.ipynb
    misc-play.ipynb
    models/
        ExtractDialogueModel.py
huggingface_token.txt
misc/
openai_api_key.txt
README.md
requirements.txt
sample_audio_clips/
tmp/
wandb/
```

## Key Components

### Models

- **ExtractDialogueModel**: This model is responsible for predicting trainee behavior and classifying feedback components. It uses OpenAI's GPT models to analyze dialogue and categorize feedback into various types such as verbal acknowledgment, behavioral change, anatomic, procedural, and technical feedback.

### Notebooks

- **analysis-analyze_dialogue_metrics.ipynb**: Analyzes dialogue metrics.
- **analysis-analyze_temporal_metrics.ipynb**: Analyzes temporal metrics.
- **analysis-confusion_matrices.ipynb**: Generates confusion matrices for model evaluation.
- **analysis-test_similarity_thresh.ipynb**: Tests similarity thresholds for feedback classification.
- **demo-dialogue_from_clip.ipynb**: Demonstrates dialogue extraction from audio clips.
- **exps-dialogue.py**: Experiments with dialogue classification.
- **exps-temporal_detection_unseen_case.py**: Experiments with temporal detection on unseen cases.
- **exps-temporal_detection_unseen_surgeon.py**: Experiments with temporal detection on unseen surgeons.
- **misc-assign_anchors.ipynb**: Assigns anchors for feedback classification.
- **misc-example_classify_dialogue.ipynb**: Example notebook for classifying dialogue.
- **misc-play.ipynb**: Miscellaneous experiments and playbook.

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Hugging Face token

### Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key and Hugging Face token:
    - Place your OpenAI API key in 

openai_api_key.txt

.
    - Place your Hugging Face token in 

huggingface_token.txt

.

## Usage

### Running the Models

To run the `ExtractDialogueModel` for predicting trainee behavior, use the following command:

```sh
python code/models/ExtractDialogueModel.py
```

### Running the Notebooks

You can explore the various Jupyter notebooks in the 

code

 directory to analyze dialogue metrics, test similarity thresholds, and more. For example, to run the `analysis-analyze_dialogue_metrics.ipynb` notebook:

```sh
jupyter notebook code/analysis-analyze_dialogue_metrics.ipynb
```

## Contributing

We welcome contributions to this project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or inquiries, please contact [your_email@example.com](mailto:your_email@example.com).

---

This README provides a comprehensive overview of the project, including its structure, key components, setup instructions, usage guidelines, and contribution process.

firdavsn: Add this to the README

Workspace: Sure, here is the updated README with the additional information:
