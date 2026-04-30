# ET-COME

## Equilibrium Transport with Conformal Minority Expansion

This repository hosts supplementary material related to the ET-COME paper. It provides both the algorithm implementation and the experimental results conducted during our research.

## Overview

- **Algorithm Implementation:**
    The core algorithm is implemented in Python using NumPy and scikit-learn. It is intended for research and experimental purposes, not for production use.

- **Experiments:**
    Extensive testing and interpretation experiments are available in the dedicated experiments folder.

- **Research Paper:**
    The accompanying research paper details the methodology and results. You can access it directly from the repository.

## Repository Structure

- **[et_come](et_come.py):**
    Contains the main algorithm implementation.

- **[experiments](./experiments/):**
    Directory with experimental scripts and data for interpretation and performance evaluation.

- **[paper](./paper/paper.pdf):**
    Includes the ET-COME paper in PDF format.


# How to run our experiments

```bash
# advanced_additional_datasets_small
python bagging_smote.py --experiment-name advanced_additional_datasets_small --only-small-datasets --only-additional-datasets --verbose --n-takes 5 --advanced

# advanced_ignore_additional_datasets_small
# here 295-301 are skipped due to data error
python bagging_smote.py --experiment-name advanced_ignore_additional_datasets_small --only-small-datasets --ignore-additional-datasets --verbose --n-takes 5 --advanced

# base_additional_datasets_small
python bagging_smote.py --experiment-name base_additional_datasets_small --only-small-datasets --only-additional-datasets --verbose --n-takes 5 --base

# base_ignore_additional_datasets_small
python bagging_smote.py --experiment-name base_ignore_additional_datasets_small --only-small-datasets --ignore-additional-datasets --verbose --n-takes 5 --base
```


## License

This project is provided for research purposes. Refer to the repository's license file for more details.

For any inquiries, feel free to open an issue or contact the repository maintainers.
