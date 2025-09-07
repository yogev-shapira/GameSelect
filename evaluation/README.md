## 📂 Evaluation

This folder contains all files used to test and validate the recommendation system through two fan surveys.

### 🔹 Python scripts
- **analyze_responses.py** — maps raw survey text responses to structured `game_id`s for evaluation.  
- **evaluate_responses.py** — computes Recall@k and NDCG@k metrics for the first survey.  
- **evaluate_responses2.py** — same as above, applied to the second survey.  
- **compare_evaluations.py** — compares evaluation outputs across methods and survey runs.  
- **run_simple.py** — quick script for running baseline or simplified evaluations.

### 🔹 Survey data
- **survey_responses.csv** — raw responses from the first fan survey.  
- **survey_responses2.csv** — raw responses from the second fan survey.  
- **survey_responses_with_ids.csv** — processed first survey (responses mapped to `game_id`s).  
- **survey_responses_with_ids2.csv** — processed second survey (mapped to `game_id`s).

### 🔹 Evaluation outputs
- **compare_methods.csv** — comparison of different recommendation methods (survey 1).  
- **compare_methods2.csv** — comparison of methods (survey 2).  
- **compare_methods_comb.csv** — combined comparison across both surveys.  
- **evaluation_metrics.csv** — computed metrics for survey 1 (main method).  
- **evaluation_metrics2.csv** — computed metrics for survey 2 (main method).  
- **evaluation_metrics_exc.csv** — evaluation using excitement-only baseline (survey 1).  
- **evaluation_metrics_exc2.csv** — same, for survey 2.  
- **evaluation_metrics_max.csv** — evaluation using max-similarity baseline (survey 1).  
- **evaluation_metrics_max2.csv** — same, for survey 2.  
- **evaluation_metrics_rnd.csv** — random baseline metrics (survey 1).  
- **evaluation_metrics_rnd2.csv** — random baseline metrics (survey 2).  
- **metrics_comparison.svg** — visualization comparing metrics across methods and surveys.  

