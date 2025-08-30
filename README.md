This project investigates a fundamental cognitive ability: the integration of information over time. The core scientific question is: **How do different neural network architectures succeed or fail at maintaining and recalling information across a temporal delay, and what can this tell us about the mechanisms of working memory?**

This work serves as a foundational experiment in computational cognitive science, where neural networks are not merely used as engineering tools, but as simplified, explorable "model organisms" for understanding the principles of cognition. The initial inspiration for this line of inquiry stems from exploring the requirements of complex theories of mind, such as Integrated Information Theory (IIT), which presuppose a system's ability to form a stable, integrated state over time.

## Experiment : Memory Dependent Response

To probe this question, a classic delayed-response task was designed and implemented:

*   **Objective:** The model is shown a discrete signal ('A' or 'B') at the very first time step of a sequence.
*   **Delay:** The signal is followed by a period of "silence," where the input is neutral (zeros).
*   **Recall:** After the delay, the model's task is to output the original signal it saw at the beginning.

Success on this task is impossible without a functioning **working memory**. The model must create and sustain an internal representation of the initial signal that is robust enough to survive the distraction-free delay period.

## Procedure & Findings

The investigation was conducted in two primary phases, yielding a critical and insightful finding.

### RNN

*   **Hypothesis:** A standard `nn.RNN` layer, the simplest form of recurrent architecture, would be capable of solving this seemingly simple task.
*   **Observation:** The simple RNN model **failed to learn effectively**. The training loss stagnated around `0.693`, which is equivalent to random guessing. Only after an exceptionally long training period (~1300 epochs) did the model occasionally and unreliably find a solution.
*   **Analysis:** This behavior is a classic symptom of the **vanishing gradient problem**. The simple RNN's mechanism for updating its memory "blends" new inputs with old memory in a way that dilutes information over time. The initial signal was effectively "washed out" by the subsequent silent inputs, leaving the optimizer with no informative gradient to guide learning. This created a "flat loss landscape" where the model could not find a path to the correct solution.

### LSTM Network

*   **Hypothesis:** A more sophisticated architecture, the `nn.LSTM`, which was specifically designed to combat the vanishing gradient problem, would learn the task more efficiently and reliably.
*   **Observation:** The LSTM-based model **succeeded dramatically**. It began learning almost immediately, with the training loss decreasing steadily and rapidly. It achieved near-perfect accuracy in under 200 epochs.
*   **Analysis:** The LSTM's success is attributable to its **gating mechanism**. This internal architecture acts as an intelligent controller of its memory. The LSTM learned to:
    1.  **"Store"** the initial signal using its *input gate*.
    2.  **"Protect"** this signal from being overwritten during the delay period by closing its *forget gate* and *input gate*.
    3.  **"Recall"** the signal for the final decision using its *output gate*.

    This architectural difference fundamentally reshaped the "loss landscape," making it a smooth, easily navigable valley and rendering the learning process robust and efficient.

## Technical Implementation

The project is structured in a modular and reproducible format:

*   **`model.py`:** A blueprint defining the neural network architectures (both the initial RNN and the final LSTM).
*   **`data_loader.py`:** A function to procedurally generate the delayed-response task data.
*   **`train.py`:** A reusable function encapsulating the core training loop.
*   **`run_experiment.ipynb`:** A Jupyter Notebook that acts as the main "lab bench" for configuring and executing the experiment.
*   **Experiment Tracking:** All experimental runs, including hyperparameters, learning curves, and final accuracy, are logged using **MLflow**. The results can be viewed and compared by running `mlflow ui` from the project's root directory.

## Next step

Analyze LSTM hidden state time series , and compute time integration 
