# TakeMeter: r/nba Discourse Classifier

## Project Overview
The TakeMeter is a task-specific Natural Language Processing (NLP) text classification model engineered to analyze public discourse within the `r/nba` subreddit community. The model categorizes noisy, fan-generated comment streams into three distinct structural and quality-based tiers:
1. **`analysis`**: Arguments and breakdowns anchored by structural basketball mechanics, specific contract rules, or verifiable statistics.
2. **`hot_take`**: Highly confident assertions and extreme opinions operating purely on narrative, casual bias, or emotion without substantial evidence.
3. **`reaction`**: Low-effort conversational filler, emotional outbursts, jokes, or community-specific memes.

By filtering technical sports evaluation out from the surrounding sea of reactionary hyperbole, this system functions as the primary backend engine for customizable subreddit feed filters and automated moderation tools.

---

## Baseline Model Evaluation

Before training our task-specific pipeline, a zero-shot baseline was established using the state-of-the-art **Llama-3.3-70b-Versatile** model via the Groq API. This evaluation was run against the locked test partition (representing 15% of the total dataset, consisting of 39 completely unique examples) to measure the intrinsic linguistic difficulty of our taxonomy.

### Baseline Performance Metrics
The zero-shot baseline processed 100% of the test queries successfully, yielding an unparseable response rate of **0.0%**. The performance metrics are summarized below:

* **Overall Baseline Accuracy:** 74.4% (29 out of 39 predictions correct)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`analysis`** | 0.71 | 0.79 | 0.75 | 19 |
| **`hot_take`** | 0.56 | 0.50 | 0.53 | 10 |
| **`reaction`** | 1.00 | 0.90 | 0.95 | 10 |
| *Macro Average* | *0.76* | *0.73* | *0.74* | *39* |
| *Weighted Average* | *0.75* | *0.74* | *0.74* | *39* |

### Reflection & Error Analysis
The baseline model excelled at identifying conversational noise and fan reactions, achieving a flawless **1.00 Precision** on the `reaction` class. However, the primary bottleneck was the **`hot_take`** label, where it collapsed to a poor **0.50 Recall** (missing exactly half of the true hot takes). This proved that the zero-shot model relied too heavily on surface-level text features like formatting correctness and an authoritative tone, falling victim to the **"Decorative Stat"** edge case where long narrative rants look professional but lack empirical substance.

---

## Fine-Tuned Model Evaluation (DistilBERT)

### Hyperparameter Optimization Journey
The Colab notebook guidelines note that standard configuration defaults are meant to act as starting baselines for small datasets (100–500 examples). Because our training dataset natively skewed toward long-form content and had a slight class imbalance (128 `analysis` vs. ~65 for the others), the model required several targeted configuration changes to learn effectively.

#### Iteration 1: The Underfitting Defaults
Initially, the model was executed with standard default hyperparameters: `num_train_epochs=3`, `learning_rate=2e-5`, and `warmup_steps=50`. This run resulted in a poor validation landscape and low test performance:
* **Initial Test Accuracy:** 56.4%
* **The Failure Mode:** The model suffered from complete majority class collapse. It generated a 1.00 recall for `analysis` but collapsed to an absolute **0.00 Recall** on `hot_take`. Because the entire run completed in only 36 steps, the model never surpassed the 50 hardcoded warmup steps—meaning it timed out before it ever reached its maximum learning rate.

#### Iteration 2: Optimization and Final Configuration
To push the model out of this local minimum, I systematically altered the training parameters. Here is exactly what was changed and why:

* **`num_train_epochs=8` (Increased from 3):** A tiny dataset of ~250 examples means there are very few weight updates per epoch. Increasing the training length gave the model enough time and passes over the data to actually learn the subtle structural boundaries of the minority classes, rather than just memorizing the easiest probability (the majority class).
* **`learning_rate=4e-5` (Increased from 2e-5):** In Iteration 1, the model was stuck in a local minimum where guessing `analysis` yielded the lowest effort-to-loss ratio. Doubling the learning rate created a stronger gradient signal, effectively "shocking" the model out of that trap and forcing it to update its weights to recognize the harder features of `hot_take` and `reaction`.
* **`warmup_ratio=0.1` (Changed from `warmup_steps=50`):** With our small dataset, 50 steps exceeded the entire duration of the training run, meaning the model never actually reached its target learning rate. Switching to a dynamic 10% ratio ensured the model safely ramped up its learning rate early on without starving the rest of the training phase.

These interventions allowed the model to escape the majority class trap, successfully beating both the initial iteration and the 70B parameter baseline.

### Final Fine-Tuned Performance Metrics
* **Overall Fine-Tuned Accuracy:** 76.9% (30 out of 39 predictions correct)
* **Improvement over Baseline:** +2.56%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`analysis`** | 0.76 | 0.84 | 0.80 | 19 |
| **`hot_take`** | 0.64 | 0.70 | 0.67 | 10 |
| **`reaction`** | 1.00 | 0.70 | 0.82 | 10 |
| *Macro Average* | *0.80* | *0.75* | *0.76* | *39* |
| *Weighted Average* | *0.79* | *0.77* | *0.77* | *39* |

### Text-Based Confusion Matrix
The text matrix below maps the raw classification counts corresponding directly to our tracked output file `confusion_matrix.png`:

| True \ Predicted | `analysis` | `hot_take` | `reaction` |
| :--- | :---: | :---: | :---: |
| **`analysis`** | **16** | 3 | 0 |
| **`hot_take`** | 3 | **7** | 0 |
| **`reaction`** | 2 | 1 | **7** |

---

## Deep Dive Error Analysis

### 1. Which labels are being confused?
Looking closely at the matrix, the most dominant error pattern is a perfectly symmetrical confusion between **`analysis`** and **`hot_take`** (3 true analysis comments predicted as hot takes; 3 true hot takes predicted as analysis). Additionally, the model struggled with `reaction` recall, accidentally misclassifying 2 reactions into `analysis` and 1 reaction into `hot_take`. Crucially, nothing was ever falsely predicted as a `reaction` (giving it a flawless 1.00 Precision).

### 2. Why is that boundary hard?
This boundary is notoriously thin because of the friction between *assertive tone* and *empirical structure*. 
* **The `analysis` ↔ `hot_take` boundary** breaks down when a user provides highly sophisticated, numberless tactical basketball breakdowns (which the model flags as a hot take due to the lack of numbers), or when a user wraps a sweeping casual generalization around a single arbitrary percentage (which tricks the model into seeing empirical analysis).
* **The `reaction` ↔ `analysis` boundary** gets blurry when a fan transitions from a low-effort joke into a multi-sentence observation. The model over-indexes on token density and length as proxies for analytical effort.

### 3. Is this a labeling problem or a data/prompt problem?
This is primarily a **data distribution and feature shortcut problem**. The human annotation was applied consistently based on our rule that true analysis *must* detail verifiable mechanics or objective metrics. However, because our raw dataset naturally had more `analysis` text blocks, the smaller DistilBERT model leaned on word length, paragraph complexity, and numeric strings as shortcuts rather than parsing the semantic intention of the poster. 

### 4. What would need to change to fix it?
To solve these remaining 9 errors, the dataset requires target expansion:
* We need more examples of **short, metric-heavy hot takes** to break the model's reliance on the "numbers = analysis" heuristic.
* We must include **long, paragraph-length casual rants** in the `reaction` class to decouple post length from true analytical depth.

---

### In-Depth Analysis of 3 Specific Failures

Here are three real examples the fine-tuned model misclassified:

**Failure 1: The Numberless Tactical Breakdown**
* **Text:** *"The Lakers' drop coverage was getting exposed all night. Reaves kept getting caught on the screen, forcing AD to step up, which left the lob wide open for Gordon."*
* **True Label:** `analysis` | **Predicted:** `hot_take`
* **Analysis:** This comment is excellent, qualitative basketball evaluation mapping defensive Pick-and-Roll adjustments. However, it completely lacks numeric data or statistics. Because the model learned a strong association between numbers and analysis during training, it rejected the structural evaluation and defaulted to `hot_take`.

**Failure 2: The Decorative Metric**
* **Text:** *"Yeah but Brunskn grew up playing against 98% right handed people. That's what makes lefties harder to deal with, you aren't as used to it. So no"*
* **True Label:** `hot_take` | **Predicted:** `analysis`
* **Analysis:** The user is making a casual, non-empirical argument about hand dominance. However, because they included a specific percentage token ("98%"), DistilBERT's attention layers over-indexed on the metric, allowing a surface-level feature to mask what was fundamentally a casual opinion.

**Failure 3: Length as a False Proxy for Quality**
* **Text:** *"As a spurs fan, don’t need him to control the game as you say, just hold the ball really lol.All jokes aside, fox has been in a different role all year and he has done a great job all things considere..."*
* **True Label:** `reaction` | **Predicted:** `analysis`
* **Analysis:** This is conversational, multi-sentence rambling containing slang ("lol"). In our training set, reactions were mostly short one-liners. Because this post was long and dense, the model assumed high-effort structure and incorrectly assigned it to `analysis`.

---

## Sample Classifications

Below are live test examples demonstrating how the fine-tuned model maps boundaries at varying confidence levels:

| Text Snippet | Predicted Label | Confidence | Real-World Validity Analysis |
| :--- | :--- | :---: | :--- |
| *"The Celtics spamming the 5-out offense against Dallas completely neutralized their rim protection..."* | **`analysis`** | **0.84** | **Correct:** The model correctly identified schematic tactical adjustments ("5-out offense") as a baseline for analysis. |
| *"Anthony Edwards is already better than Dwyane Wade ever was. Wade played with Shaq..."* | **`hot_take`** | **0.88** | **Correct:** Accurately captured a confident legacy comparison stated without statistical backing. |
| *"Just fell to my knees in a Walmart parking lot."* | **`reaction`** | **0.96** | **Correct:** High confidence mapping to a highly standardized, low-effort internet meme template. |

---

## Reflection: Model vs. Intended Capture

**What I intended to capture:** I wanted to establish a rigorous semantic boundary separating objective, verifiable basketball mechanics (`analysis`) from narrative-driven opinions (`hot_take`).

**What the model actually captured:** The model optimized for surface-level shortcuts. By tracking changes across hyperparameters, we prevented full class collapse, but the model still heavily weights structural markers—if it sees a numeric symbol, it maps toward `analysis`; if it sees multiple clauses or long sentences, it maps away from `reaction`. The model successfully learned our taxonomy, but it evaluates structural formatting and linguistic density rather than deep basketball reasoning.

---

## Spec Reflection

* **How the Spec Guided Implementation:** Pre-defining the "Decorative Stat" boundary rule in `planning.md` was vital. When the model fell into the trap of misclassifying the "98% right handed" comment, the spec gave me the exact analytical framework to diagnose the failure instead of treating it as a random glitch.
* **How Implementation Diverged from the Spec:** My data collection spec called for scraping and adding more text if a class was underrepresented. Instead, when faced with extreme underfitting and majority class bias in early testing, I adjusted via model architecture—boosting the learning rate to `4e-5` and extending the training length to 8 epochs to force the network to prioritize minority patterns.

---

## AI Usage Disclosure

1. **Dataset Augmentation Assistance:** An LLM was utilized to pre-label raw data feeds and generate additional messy text templates for the minority `reaction` class. *Override:* The AI initially generated strings that felt structurally predictable. I manually edited several lines to introduce natural typos and erratic syntax to match true subreddit formatting.
2. **Failure Analysis Pattern Recognition:** I provided my misclassified test pairs to an LLM to identify themes across our errors. *Override:* The AI suggested that total sentence count was the primary failure point. I cross-checked this against the dataset, verifying that the model was actually tripping over the combination of phrase length mixed with arbitrary percentages, allowing me to refine the final report with precise conclusions.