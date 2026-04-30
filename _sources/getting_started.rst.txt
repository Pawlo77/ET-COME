Getting Started
===============

Installation
------------

From PyPI (recommended)::

    pip install et-come

From source::

    git clone https://github.com/Pawlo77/ET-COME.git
    cd ET-COME
    make install

Quick Start Example
-------------------

Basic usage with scikit-learn datasets:

.. code-block:: python

    from src.et_come import ET_COME
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    # Create an imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create and fit ET-COME
    et_come = ET_COME(
        classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        iterations=5,
        random_state=42
    )
    et_come.fit(X_train, y_train)

    # Make predictions
    y_pred = et_come.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))

Key Parameters
--------------

- **classifier**: Base classifier to use in the ensemble
- **iterations**: Number of equilibrium iterations
- **random_state**: Random seed for reproducibility
- **q_e**: Epistemic uncertainty threshold for admissibility
- **q_a**: Aleatoric uncertainty threshold for admissibility
- **k**: Number of neighbors for HNSW graph and transport support

Next Steps
----------

- Explore the :doc:`documentation` for complete API reference
- Check out the experiments in the `experiments/` directory
- Review the paper for methodological details
- See :doc:`additional_notes` for implementation considerations
