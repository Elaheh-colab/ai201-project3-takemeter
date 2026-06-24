# TakeMeter Project Spec & Planning

For this project, I am analyzing the r/nba subreddit, a highly active basketball community where daily discourse ranges from deep statistical breakdowns to raw, emotional fan venting. The model classifies comments into three distinct categories—analysis (evidence-based arguments), hot_take (bold, unbacked assertions), and reaction (purely emotional or meme-based responses). This taxonomy matters deeply to the community because it helps users separate genuine, insightful basketball evaluation from the overwhelming noise of reactionary hyperbole, narrative-driven biases, and thought-terminating clichés.

---

## 1. Milestone 1 Checkpoint

### Label Definitions and Examples

* **`analysis`**: A comment that makes a structured argument backed by specific, verifiable basketball evidence, advanced statistics, or tactical observations.
    * *Example 1:* "The Celtics spamming the 5-out offense against Dallas completely neutralized their rim protection. By pulling Lively to the perimeter, it allowed Tatum and Brown to drive baseline with zero weak-side help."
    * *Example 2:* "Over the last month of basketball, OKC's Jalen Williams has averaged 21.5/5.1/4.1 on 56% FG, and 63.1% TS. The Thunder are 33-1 without him over two seasons, his efficiency is completely absurd."
* **`hot_take`**: A bold, confident opinion or assertion stated without substantial supporting evidence, often relying heavily on narrative, emotion, or thought-terminating clichés.
    * *Example 1:* "Giannis is wildly overrated and is closer to being out of the top 40 all-time than top 25. All he does is run and dunk, if you build a wall he's completely neutralized."
    * *Example 2:* "Anthony Edwards is already better than Dwyane Wade ever was. Wade played with Shaq and LeBron, Ant is carrying a poverty franchise by himself."
* **`reaction`**: An immediate emotional response, meme, or joke containing little to no actual argument, analysis, or definitive claim about the sport.
    * *Example 1:* "Just fell to my knees in a Walmart parking lot."
    * *Example 2:* "LMAOOOO the Suns really traded their entire future for KD just to get swept by Anthony Edwards."

---

## 2. Hard Edge Cases & Ambiguity Rules

During the annotation process, some comments sat squarely on the boundary between two labels. Here are three specific, difficult examples from the dataset and the decision rules applied to resolve them:

**Example 1: The "Tactical but Numberless" Argument (`analysis` vs. `hot_take`)**
* *Text:* "The Knicks defense has been built to cover for teams hunting mismatches. They are really good at swarming the paint when there is a breakdown and recovering to good shooters when the opponents try and pass out."
* *The Difficulty:* This comment contains zero statistics, which initially makes it look like an opinionated hot take.
* *The Decision:* **`analysis`**. I established a rule that an argument does not need numbers to be analytical; it just needs *verifiable mechanics*. Because this post details specific tactical movements (swarming the paint, recovering to shooters), it is a structural evaluation, not a narrative hot take.

**Example 2: The "Angry Observation" (`hot_take` vs. `reaction`)**
* *Text:* "I'm literally throwing up rn, how do you blow a 20 point lead in the 4th quarter. Fire Darvin Ham into the sun."
* *The Difficulty:* The comment mentions a specific in-game event (blowing a 20-point lead) and a specific coach, which borders on making a basketball claim.
* *The Decision:* **`reaction`**. The core of the post is purely emotional venting ("throwing up rn," "fire into the sun"). Even though it references a stat (20 points), there is no argument being made, only visceral frustration.

**Example 3: The "Narrative with Math" (`hot_take` vs. `analysis`)**
* *Text:* "For example, you brought up the highest FTA by a guard in NBA history. That's great and all but the [estimated] pace during the 60s was so absurd, it's not even comparable to any point in NBA history. Using the year he had a career high in FTA, we saw a pace of 121.4 possessions on average."
* *The Difficulty:* The user is aggressively tearing down a historical player's legacy, which is classic `hot_take` behavior. 
* *The Decision:* **`analysis`**. Even though the tone is argumentative, the user grounds their claim in a highly specific, verifiable calculation (121.4 possessions average pace). The math is the foundation of the comment, earning it the analysis label.

---

## 3. Data Collection Plan & Execution

* **Data Source:** Examples were pulled from `r/nba` and its dedicated analysis offshoot `r/nbadiscussion` to capture a balanced cross-section of text lengths.
* **Target Distribution:** The goal was a dataset of at least 200 examples, aiming for an optimal balance of roughly 65+ examples per label to avoid class-imbalance biases.
* **Execution & Merging:** Three separate data sources were consolidated, deduplicated by text string, and randomized. The final processed dataset resulted in **259 unique rows**. The final distribution naturally skewed slightly toward the longest-form posts: `analysis` (128), `reaction` (66), and `hot_take` (65).

---

## 4. Evaluation Metrics

To objectively evaluate both our fine-tuned DistilBERT model and the Groq/Llama baseline, I tracked three key metrics:
1.  **Overall Accuracy:** To give a high-level view of how many total test samples were classified correctly.
2.  **Per-Class Precision:** Crucial for the `analysis` class to ensure that when our TakeMeter tags a post as highly substantive, it is not accidentally letting a high-effort `hot_take` slip through.
3.  **Per-Class Recall:** Crucial for the `reaction` label to ensure we are filtering out all the conversational noise and meme spam successfully.

---

## 5. Definition of Success

* **Deployment Standard:** For a highly subjective internet text classification task with a smaller footprint of target rows, an overall accuracy of **$\ge$ 75%** on the hold-out test split will be considered a deployment success. 
* **Real-World Utility:** A model performing at this tier is genuinely useful for a community tool (such as an automated subreddit feed filter), as it cleanly separates dense technical discussions from conversational reactions and surface-level sports arguments.

---

## 6. AI Tool Plan

* **Label Stress-Testing:** Before committing to full annotation, I used an LLM to generate highly ambiguous, borderline posts sitting at the boundary between `analysis` and `hot_take`. This exposed "label drift" (e.g., qualitative opinions masquerading as data-backed analysis) and forced me to tighten my taxonomic definitions.
* **Annotation Assistance:** I used an LLM to pre-label a batch of raw examples to accelerate the dataset creation. I manually reviewed, corrected, and verified the LLM's baseline predictions line-by-line against my strict decision boundaries to establish the human-in-the-loop ground truth.
* **Failure Analysis:** Post-evaluation, I extracted misclassified text pairs and used an LLM for pattern extraction. This helped identify systemic error configurations (like the "Decorative Stat" trap) which were then independently verified via confusion matrix analysis.

---

## 7. Mid-Project Iteration & Design Decisions

During the actual fine-tuning process (Milestone 5), I encountered a critical issue that required strategic intervention: **The Majority Class Trap**.

* **The Problem:** Because the dataset natively contained 128 `analysis` rows but only 65 `hot_take` rows, early fine-tuning runs resulted in the model predicting `analysis` too aggressively. In iteration 1, the model achieved a 1.00 recall for `analysis` but completely collapsed on `hot_take` (recall 0.20). DistilBERT learned the mathematical probability that an ambiguous post was statistically more likely to be `analysis` and stopped trying to find the boundary.
* **The Solution:** Rather than throwing away 63 natural `analysis` rows to forcibly balance the dataset (which would reduce the overall learning volume below 200 rows), I elected to intervene via hyperparameter tuning. I increased the learning rate from `2e-5` to `4e-5` and adjusted the warmup ratio to `0.1` (10% of total steps). This stronger learning signal successfully forced the model out of its local minimum, allowing it to correctly identify the harder `hot_take` patterns while maintaining the natural distribution of the dataset.