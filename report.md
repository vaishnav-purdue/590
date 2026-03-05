# Marketing Campaign Analytics — Project Report

**Course:** DSCI 590 | **Dataset:** Nykaa, Purplle & Tira Campaign Data  
**Pipeline:** PySpark 4.0.1 + MLflow | **Total Records:** 166,665

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Data Ingestion](#2-phase-1--data-ingestion)
3. [Phase 2 — Silver Layer: Cleaning & Feature Engineering](#3-phase-2--silver-layer-cleaning--feature-engineering)
4. [Phase 3 — Exploratory Data Analysis (EDA)](#4-phase-3--exploratory-data-analysis-eda)
5. [Phase 4 — Gold Layer & Predictive Modeling](#5-phase-4--gold-layer--predictive-modeling)
6. [Phase 5 — MLflow Experiment Tracking](#6-phase-5--mlflow-experiment-tracking)
7. [Insights & Business Recommendations](#7-insights--business-recommendations)
8. [Challenges & Possible Improvements](#8-challenges--possible-improvements)

---

## 1. Project Overview

This project builds an end-to-end marketing analytics pipeline for three major Indian beauty and cosmetics e-commerce brands: **Nykaa**, **Purplle**, and **Tira**. The pipeline follows a medallion architecture (Bronze → Silver → Gold), performs comprehensive exploratory data analysis, and culminates in a machine learning classifier that predicts whether a campaign will fall in the **top 25% by ROI** — before the campaign launches.

**Core business question:**  
> Given only pre-campaign attributes (channel, campaign type, customer segment, language, duration, timing), can we predict which campaigns will be high-ROI?

**Tech stack:** PySpark 4.0.1, MLflow, Pandas, Matplotlib, Seaborn

---

## 2. Phase 1 — Data Ingestion

### Data Sources

Three separate CSV files were ingested, one per brand:

| File | Brand |
|------|-------|
| `data/nykaa_campaign_data.csv` | Nykaa |
| `data/purplle_campaign_data.csv` | Purplle |
| `data/tira_campaign_data.csv` | Tira |

### Ingestion Pipeline

1. **Schema normalization** — Column names converted to lowercase with non-alphanumeric characters replaced by underscores, ensuring consistency across all three sources.
2. **Date parsing** — Two date formats handled automatically: `dd-MM-yyyy` and `MM/dd/yyyy`.
3. **Brand tagging** — A `brand_source` column was added to each row to preserve provenance after the union.
4. **Schema alignment** — Missing columns across DataFrames were filled with `null` before union via `unionByName`.
5. **Deduplication** — Exact duplicate rows removed with `dropDuplicates()`.
6. **Spark session** — Initialized with 8 GB driver memory and 8 shuffle partitions for local execution.

### Result

| Metric | Value |
|--------|-------|
| Total rows after ingestion | **166,665** |
| Number of brands | 3 |
| Raw columns | 16 |

### Dataset Schema (Raw)

| Column | Type | Description |
|--------|------|-------------|
| `campaign_id` | string | Unique campaign identifier |
| `campaign_type` | string | Email, Social Media, Influencer, Paid Ads, SEO |
| `channel_used` | string | One or more channels (comma-separated) |
| `impressions` | long | Total ad impressions |
| `clicks` | long | Total ad clicks |
| `leads` | long | Leads generated |
| `conversions` | long | Conversions achieved |
| `acquisition_cost` | double | Total spend on campaign |
| `revenue` | long | Revenue attributed to campaign |
| `roi` | double | Return on investment |
| `engagement_score` | double | Composite engagement metric |
| `duration` | long | Campaign duration in days |
| `customer_segment` | string | Premium Shoppers, College Students, Youth, etc. |
| `target_audience` | string | Audience targeting label |
| `language` | string | Language of campaign (e.g., Bengali, Tamil) |
| `date` | timestamp | Campaign date |

---

## 3. Phase 2 — Silver Layer: Cleaning & Feature Engineering

### Data Cleaning

| Step | Action |
|------|--------|
| Null strings | Replaced with `"Unknown"` |
| Whitespace | Trimmed from all string columns |
| Negative numeric values | Replaced with `null` for non-negative columns (clicks, impressions, conversions, revenue, etc.) |
| ROI negatives | Retained (negative ROI is a valid business outcome) |

### Engineered KPI Features

Eight new KPI columns were derived to capture funnel efficiency and return metrics:

| KPI | Formula | Business Meaning |
|-----|---------|-----------------|
| `ctr` | `clicks / impressions` | Click-through rate; ad creative effectiveness |
| `lead_rate` | `leads / clicks` | Landing page / top-of-funnel conversion |
| `lead_to_conv` | `conversions / leads` | Lead qualification quality |
| `click_to_conv` | `conversions / clicks` | Full funnel conversion rate |
| `cpc` | `acquisition_cost / clicks` | Cost per click |
| `cpa` | `acquisition_cost / conversions` | Cost per acquisition |
| `roas` | `revenue / acquisition_cost` | Return on ad spend |
| `rpm` | `(revenue × 1000) / impressions` | Revenue per thousand impressions |

All ratios use safe division (null returned when denominator ≤ 0). Funnel ratios (CTR, lead_rate, lead_to_conv, click_to_conv) are clipped to [0, 1].

### Silver Layer Output

- **166,665 rows × 25 columns** saved as Parquet at `data/marketing_silver_parquet`
- Registered as a Spark SQL temporary view (`marketing_silver`)

---

## 4. Phase 3 — Exploratory Data Analysis (EDA)

### 4.1 Overall Summary Statistics

| Metric | Value |
|--------|-------|
| Total campaigns | **166,665** |
| Average impressions | **55,060.85** |
| Average clicks | **4,682.37** |
| Average conversions | **1,029.09** |
| Average CTR | **8.50%** |
| Average ROI | **2.69** |

### 4.2 Full Numeric Distribution

| Column | Min | 25th pct | Median | Mean | 75th pct | Max |
|--------|-----|----------|--------|------|----------|-----|
| `acquisition_cost` | 8.18 | 106.71 | 208.47 | 376.09 | 427.40 | 15,473.16 |
| `clicks` | 202 | 2,108 | 3,903 | 4,682 | 6,687 | 14,944 |
| `conversions` | 17 | 401 | 776 | 1,029 | 1,403 | 6,686 |
| `duration` | 5 | 11 | 17 | 17.49 | 24 | 30 |
| `engagement_score` | 2.56 | 8.38 | 13.59 | 13.77 | 18.79 | 30.99 |
| `impressions` | 10,001 | 32,561 | 55,102 | 55,061 | 77,565 | 100,000 |
| `leads` | 48 | 779 | 1,475 | 1,872 | 2,598 | 8,876 |
| `revenue` | 3,895 | 177,660 | 359,172 | 513,907 | 684,632 | 4,579,910 |
| `roi` | -0.99 | 0.04 | 1.23 | 2.69 | 3.58 | 79.30 |
| `ctr` | 0.020 | 0.053 | 0.085 | 0.085 | 0.118 | 0.150 |
| `lead_rate` | 0.198 | 0.299 | 0.399 | 0.399 | 0.499 | 0.600 |
| `lead_to_conv` | 0.288 | 0.424 | 0.550 | 0.550 | 0.674 | 0.800 |
| `click_to_conv` | 0.059 | 0.152 | 0.205 | 0.219 | 0.277 | 0.479 |
| `cpc` | 0.001 | 0.017 | 0.052 | 0.320 | 0.189 | 59.51 |
| `cpa` | 0.001 | 0.078 | 0.264 | 2.076 | 1.027 | 870.73 |
| `roas` | 0.259 | 444.52 | 1,761.82 | 6,439.07 | 6,112.35 | 504,395.37 |
| `rpm` | 299.73 | 4,221.18 | 7,457.84 | 9,331.38 | 12,513.86 | 51,321.10 |

### 4.3 Key EDA Observations

**ROI Distribution**
- The ROI distribution is **right-skewed** (mean 2.69, median 1.23), indicating a small number of exceptionally high-performing campaigns pulling the average upward.
- Only ~25.7% of campaigns achieve ROI ≥ 3.48 (the 75th percentile threshold used as the Gold label).
- At the extreme tail, some campaigns achieve an ROI as high as **79.3×**.

**Channel Performance**
- Campaigns were distributed across channels: Facebook, Google, Email, Instagram, WhatsApp, and YouTube (often multi-channel).
- **YouTube** (rank 2, importance 0.043) and **Instagram** (rank 3, importance 0.034) are the highest-ranked channel features in the final GBT model.
- **Email** (rank 7, importance 0.030) also appears, while Google and WhatsApp did not appear in the top 10 for the tree-based model.

**Campaign Type Performance**
- Analysis was grouped by `campaign_type` and sorted by `avg_roi`.
- Campaign type features did not appear in the top 10 GBT importances, suggesting that the type of campaign is less predictive than duration, channel mix, and timing when using tree-based models.

**Brand Comparison**
- All three brands (Nykaa, Purplle, Tira) had consistent campaign structures and date ranges.
- Brand features did not appear in the top 10 GBT importances, suggesting brand alone does not strongly differentiate ROI class in the tree model.

**Customer Segment Patterns**
- Three segments appeared in the top 10 GBT importances: **Tier 2 City Customers** (rank 5, 0.032), **Working Women** (rank 8, 0.030), and **Premium Shoppers** (rank 9, 0.027).
- The presence of Tier 2 City Customers at the top suggests regional expansion campaigns may have different ROI profiles worth investigating.

**Temporal Patterns**
- Monthly ROI trends were visualized; performance varied noticeably by month.
- **February** (`month_ohe_2.0`, rank 4, importance 0.033) and **Wednesday** (`dayofweek_ohe_3.0`, rank 6, importance 0.032) are the strongest temporal split points in the GBT model.
- Note: `year` — the second-strongest feature in the earlier logistic regression analysis — does not appear in the top 10 GBT importances, highlighting how model choice affects which features surface as important.

**Engagement Score**
- Mean engagement score of **13.77** with a range of 2.56–30.99.
- Engagement score was positively correlated with conversions and ROI.

---

## 5. Phase 4 — Gold Layer & Predictive Modeling

### 5.1 Gold Label Definition

The **75th percentile ROI** was computed across all 166,665 campaigns:

> **Gold threshold: ROI ≥ 3.48**

| Label | Meaning | Count | Share |
|-------|---------|-------|-------|
| `1` | High-performing (top 25% ROI) | 42,895 | 25.7% |
| `0` | Standard ROI | 123,770 | 74.3% |

The resulting Gold parquet was saved at `data/marketing_gold_parquet`.

### 5.2 Feature Design (No-Leakage Principle)

To build a **pre-campaign predictor** (useful before launch), only attributes known at campaign planning time were included:

| Feature Type | Features |
|-------------|---------|
| Categorical (OHE) | `brand_source`, `campaign_type`, `customer_segment`, `language` |
| Numeric | `duration`, `year`, `month`, `dayofweek` |
| Channel flags (binary) | `ch_facebook`, `ch_whatsapp`, `ch_youtube`, `ch_google`, `ch_email`, `ch_instagram` |

**Pipeline:** `StringIndexer` → `OneHotEncoder` → `VectorAssembler`

> In-campaign metrics (clicks, impressions, CTR, ROAS, etc.) were deliberately excluded to prevent data leakage.

**Train / Test split:** 80/20 (133,650 train / 33,015 test), stratified by seed=42.

### 5.3 Models & Hyperparameter Grids

Three classifiers were trained using 3-fold `CrossValidator`:

| Model | Hyperparameter Grid |
|-------|-------------------|
| **Logistic Regression** | `regParam` ∈ {0.01, 0.1}, `elasticNetParam` ∈ {0.0, 0.5}, `maxIter=50` |
| **Random Forest** | `numTrees` ∈ {50, 100}, `maxDepth` ∈ {3, 5} |
| **GBT Classifier** | `maxDepth` ∈ {3, 5}, `stepSize` ∈ {0.05, 0.1}, `maxIter=20` |

### 5.4 Model Evaluation Results

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Logistic Regression | 0.4985 | 0.5672 | 0.7532 | 0.6471 |
| Random Forest | 0.4968 | 0.5672 | 0.7532 | 0.6471 |
| **GBT Classifier** | **0.4995** | 0.5672 | 0.7532 | 0.6471 |

**Best model: GBT Classifier (AUC: 0.4995)**

> The near-identical Precision/Recall/F1 across all three models indicates majority-class collapse — all three models default to predicting label=0 at similar rates. AUC near 0.50 confirms that pre-campaign features alone do not reliably separate the top 25% ROI campaigns from the rest. Feature importances from these models are exploratory in nature and should not be treated as actionable without further validation.

### 5.5 Top 10 Feature Importances (GBT Classifier)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `duration` | 0.2693 |
| 2 | `ch_youtube` | 0.0428 |
| 3 | `ch_instagram` | 0.0336 |
| 4 | `month_ohe_2.0` | 0.0328 |
| 5 | `customer_segment_ohe_Tier 2 City Customers` | 0.0324 |
| 6 | `dayofweek_ohe_3.0` | 0.0320 |
| 7 | `ch_email` | 0.0303 |
| 8 | `customer_segment_ohe_Working Women` | 0.0299 |
| 9 | `customer_segment_ohe_Premium Shoppers` | 0.0273 |
| 10 | `language_ohe_Bengali` | 0.0263 |

**Interpretation:**
- **Campaign duration** is by far the dominant feature (importance 0.27), suggesting that how long a campaign runs is the single strongest pre-campaign signal the model can leverage — though given the near-random AUC, this should be treated as a weak association rather than a reliable predictor.
- **YouTube and Instagram channels** rank 2nd and 3rd, indicating multi-channel campaigns that include video platforms are somewhat more likely to appear in the top-ROI tier.
- **Temporal features** (`month`, `dayofweek`) contribute meaningful but small signal — consistent with seasonal promotion patterns in the Indian beauty e-commerce calendar.
- **Tier 2 City Customers** and **Working Women** segments appear in the top importances, alongside **Premium Shoppers**.
- **Language (Bengali)** ranks 10th, suggesting regional targeting has some association with ROI outcomes.

---

## 6. Phase 5 — MLflow Experiment Tracking

### Experiment Setup

- **Experiment name:** `Campaign_ROI_Classification`
- Each model run logged: model name, all hyperparameters, AUC, Precision, Recall, F1, and the serialized Spark model artifact.

### Logged Runs

| Run Name | Model | AUC | F1 | Run ID (prefix) |
|----------|-------|-----|----|-----------------|
| `LR_Pipeline_CV` | Logistic Regression | 0.4985 | 0.6471 | `713fff4b...` |
| `RF_Pipeline_CV` | Random Forest | 0.4968 | 0.6471 | `77173863...` |
| `GBT_Pipeline_CV` | GBT Classifier | 0.4995 | 0.6471 | `cc49fcd2...` |
| `Model_Comparison_Summary` | — (cross-run summary) | — | — | — |

Each run logged the complete Spark ML Pipeline artifact (all preprocessing stages + classifier) so the full pipeline can be reloaded and applied to raw feature data without manual preprocessing.

### Best Model (Registered & Loaded from Registry)

The best Phase 5 model was registered to the MLflow Model Registry and loaded back for automated inference:

| Metric | Value |
|--------|-------|
| Model | GBT Classifier |
| Registry name | `CampaignROI_BestModel` |
| Version | 1 |
| Status | READY |
| Model URI | `runs:/cc49fcd2afa84c05a1af23f2b311aa1f/gbt_pipeline` |
| Loaded pipeline stages | `StringIndexerModel ×7, OneHotEncoderModel, VectorAssembler, GBTClassificationModel` |

**Automated inference (loaded pipeline applied to raw test data):**

| Metric | Value |
|--------|-------|
| AUC | 0.4995 |
| Precision | 0.5672 |
| Recall | 0.7532 |
| F1 | 0.6471 |

Metrics are identical to the original training run, confirming end-to-end pipeline serialization and reproducibility.

---

## 7. Insights & Business Recommendations

### 7.1 Key Findings from the Predictive Model

> **Important caveat:** All three models achieved AUC ≈ 0.50 with identical Precision/Recall/F1 — consistent with majority-class collapse. The findings below are exploratory observations from the GBT feature importances and should be treated as hypotheses for further investigation, not confirmed business rules.

1. **Pre-campaign features do not reliably predict top-25% ROI.** All three models converge to near-random AUC (~0.499), indicating that campaign channel, type, segment, language, and duration alone do not contain enough signal to separate high-ROI campaigns before launch. In-campaign metrics (CTR, engagement score, ROAS) would provide far stronger signal but introduce data leakage for a planning tool.

2. **Campaign duration is the strongest structural differentiator.** With a GBT importance of 0.27 — dwarfing all other features — duration is the single attribute most correlated with ROI class separation in the tree splits. This may reflect that longer campaigns accumulate more impressions and conversions, or that budget allocation policies (which co-vary with duration) are the real underlying driver.

3. **Video and social channels (YouTube, Instagram) show the next-strongest associations.** YouTube ranks 2nd (0.043) and Instagram 3rd (0.034) in GBT importance. Including these channels in campaign design is associated with higher probability of landing in the top ROI quartile, though the effect is small given the near-random overall model.

4. **Temporal patterns contribute meaningful splits.** February (`month_ohe_2.0`) and a specific day-of-week (`dayofweek_ohe_3.0`, Wednesday) appear in the top 6, suggesting that launch timing relative to seasonal demand or budget cycles has some association with ROI outcomes.

5. **Tier 2 City Customers and Working Women emerge alongside Premium Shoppers.** All three segments appear in the top 10 importances (ranks 5, 8, 9), indicating that segment targeting has modest but consistent influence on tree-based models. Premium Shoppers retains its positive association established in prior analysis.

6. **Regional language targeting (Bengali) is a marginal signal.** Its presence at rank 10 (importance 0.026) suggests regional campaigns may behave differently in aggregate, but the effect is too small to treat as actionable without more targeted analysis.

### 7.2 Actionable Business Recommendations

> These recommendations are grounded in the GBT feature importances and EDA observations. Given the near-random model AUC, they represent directional hypotheses for A/B testing — not prescriptive rules.

**1. Investigate the relationship between campaign duration and ROI**  
Duration is overwhelmingly the top GBT feature (importance 0.27). Before concluding that "longer campaigns = higher ROI", investigate whether duration co-varies with budget (larger campaigns tend to run longer), audience size, or campaign type. If the relationship is causal, it has direct budget-planning implications.

**2. Prioritize YouTube and Instagram in channel mix**  
YouTube (rank 2, 0.043) and Instagram (rank 3, 0.034) are the top channel-based features. Campaigns that include these channels in their mix appear more often in the top ROI quartile. Brands should test adding YouTube to campaigns currently running on Instagram-only or email-only to assess whether the channel combination drives incremental ROI.

**3. Time campaign launches around February and mid-week**  
February (`month_ohe_2.0`) and Wednesday (`dayofweek_ohe_3.0`) both appear as meaningful split points in the GBT. This could reflect Valentine's Day promotions, post-January restocking behavior, or mid-week purchase intent patterns. Consider A/B testing launch timing to validate.

**4. Expand targeting to Tier 2 City Customers and Working Women**  
These two segments appear in the top 10 importances alongside Premium Shoppers. They may represent underserved high-potential segments in the Indian beauty market. Brands should analyze their historical campaign ROI by segment more granularly before shifting targeting budgets.

**5. Use the MLflow pipeline as a pre-launch screening tool — with caveats**  
The registered `CampaignROI_BestModel` (GBT, version 1) can be loaded and applied to any new campaign's planned attributes to generate a high/standard ROI probability score. Given the current AUC (~0.50), the score should be treated as a weak tiebreaker, not a decision gate. The primary value is establishing a baseline for future model improvement.

**6. Retrain when in-campaign data becomes available**  
The model is intentionally limited to pre-campaign features. Once a campaign has run for 7–14 days, re-scoring with early in-campaign signals (CTR, engagement score) would produce substantially higher AUC. An MLflow-automated two-stage scoring system (pre-launch + mid-campaign) would provide far more actionable guidance.

---

## 8. Challenges & Possible Improvements

### Challenges Encountered

| Challenge | Description |
|-----------|-------------|
| **No-leakage constraint limits AUC** | Using only pre-campaign features is correct for a planning tool, but it inherently limits discriminative power. In-campaign signals (CTR, engagement score) are far more predictive — but using them would make the model useless for pre-launch decisions. |
| **Class imbalance (74.3% / 25.7%)** | The 3:1 class imbalance means models that default to the majority class achieve ~74% accuracy while providing no real insight. AUC and F1 are more meaningful here, but all three models still trend toward majority-class dominance. |
| **Identical F1/Precision/Recall across models** | All three models produced the same weighted Precision (0.5672), Recall (0.7532), and F1 (0.6471). This convergence indicates majority-class collapse — every model predicts label=0 at a similar rate, a symptom of the feature set's limited separating power against the 74/26 class imbalance. |
| **Compute constraints** | Local PySpark execution (even with 8 GB driver memory) limited the depth of hyperparameter search possible. A broader grid search might have revealed more differentiation between models. |
| **Channel multi-valency** | The `channel_used` column contains comma-separated values (e.g., `"YouTube, WhatsApp, Instagram"`). Binary flag encoding was used, but interaction effects between channels were not modeled. |

### Possible Improvements

| Improvement | Expected Impact |
|------------|----------------|
| **Add class balancing** (SMOTE or class weights) | Would force models to learn minority class patterns more aggressively, likely improving recall for high-ROI campaigns |
| **Incorporate external signals** | Festive calendar flags, competitor spend data, or macro-economic indicators could add meaningful predictive power without leakage |
| **Channel interaction features** | Creating features for multi-channel combinations (e.g., `Instagram + Influencer`) could capture synergy effects |
| **Target encoding for high-cardinality features** | Replace OHE for `campaign_id`-level features with target-encoded versions to capture more nuanced historical patterns |
| **Deep hyperparameter search on cluster** | Running the pipeline on a full Spark cluster (e.g., Databricks, EMR) would allow grid sizes 10× larger, potentially improving AUC |
| **Time-series cross-validation** | Replace random 80/20 split with temporal splits (train on older campaigns, test on newer) to better reflect real-world deployment and avoid look-ahead bias |
| **Gradient Boosting with LightGBM/XGBoost** | Native LightGBM with better handling of imbalanced data and categorical features could outperform the PySpark GBT implementation |
| **Post-hoc ROI prediction as regression** | In addition to the binary classifier, training a regression model to predict the actual ROI value would allow richer budget allocation decisions |
| **Automated retraining pipeline** | Integrating MLflow with a scheduler (e.g., Apache Airflow) for monthly model refresh would ensure the screening tool remains accurate as market conditions evolve |

---

## Summary

This project successfully built a full end-to-end marketing analytics pipeline across five phases — from raw CSV ingestion through Silver-layer cleaning, EDA, Gold-label creation, and MLflow-tracked ML modeling — for 166,665 campaign records across Nykaa, Purplle, and Tira.

The predictive models (LR, RF, GBT) all achieved AUC ≈ 0.50 with identical Precision (0.5672) / Recall (0.7532) / F1 (0.6471), confirming that pre-campaign features alone do not reliably discriminate the top-ROI quartile. The best model, **GBT Classifier** (AUC 0.4995), was registered to the MLflow Model Registry and validated via automated pipeline inference. Key GBT feature importances highlight **campaign duration, YouTube/Instagram channels, February timing, and Tier 2 / Working Women segments** as the most influential pre-campaign attributes — providing directional hypotheses for future experimentation. The pipeline is MLOps-ready (full Spark ML Pipeline artifacts logged, registered, and loadable from MLflow), with a clear roadmap for accuracy improvements through class balancing, in-campaign signals, and temporal cross-validation.

---

*Report generated from `daniel_testing.ipynb` — March 2026*
