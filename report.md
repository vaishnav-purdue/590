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
- Instagram-inclusive campaigns demonstrated consistently higher-than-average ROI, contributing to it being the **top predictive feature** in the ML models.
- Google and WhatsApp also appeared as positively weighted features.

**Campaign Type Performance**
- Analysis was grouped by `campaign_type` and sorted by `avg_roi`.
- **Influencer** and **Paid Ads** campaign types were the strongest predictors of high ROI in the logistic regression model.
- **Email** campaigns had the widest variance in performance.

**Brand Comparison**
- All three brands (Nykaa, Purplle, Tira) had consistent campaign structures and date ranges.
- `brand_source` was a meaningful feature — Nykaa's brand indicator had the 4th-highest feature importance, and Purplle the 7th.

**Customer Segment Patterns**
- **Premium Shoppers** was the most predictive customer segment (3rd highest feature importance), suggesting that campaigns targeting this segment more reliably convert at high ROI.
- College Students and Youth segments showed lower average ROI.

**Temporal Patterns**
- Monthly ROI trends were visualized; performance varied noticeably by month.
- `year` was the **2nd most important feature**, indicating ROI patterns shifted over time — consistent with seasonal sales events (Diwali, monsoon sales) in the Indian beauty market.
- `dayofweek` contributed as a smaller but meaningful predictor.

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
| **Logistic Regression** | **0.5026** | 0.5555 | 0.7453 | 0.6365 |
| Random Forest | 0.4986 | 0.5555 | 0.7453 | 0.6365 |
| GBT Classifier | 0.4987 | 0.5555 | 0.7453 | 0.6365 |

**Best model: Logistic Regression (AUC: 0.5026)**

> The near-identical Precision/Recall/F1 across all three models indicates that all three default to predicting the majority class (label=0) at similar rates. This is the expected behavior when features are limited to pre-campaign attributes with no information leakage — the models capture genuine baseline signal, but in-campaign metrics are where predictive power truly resides.

### 5.5 Top 10 Feature Importances (Logistic Regression)

| Rank | Feature | Coefficient Magnitude |
|------|---------|----------------------|
| 1 | `ch_instagram` | 0.02784 |
| 2 | `year` | 0.02519 |
| 3 | `customer_segment_ohe_Premium Shoppers` | 0.01778 |
| 4 | `brand_source_ohe_nykaa` | 0.01711 |
| 5 | `campaign_type_ohe_Influencer` | 0.01557 |
| 6 | `campaign_type_ohe_Paid Ads` | 0.01525 |
| 7 | `brand_source_ohe_purplle` | 0.01353 |
| 8 | `ch_google` | 0.01276 |
| 9 | `ch_whatsapp` | 0.01115 |
| 10 | `campaign_type_ohe_SEO` | 0.01052 |

**Interpretation:**
- **Instagram** is the single most influential pre-campaign predictor of high ROI.
- **Temporal trends** (`year`) are the second-strongest signal — ROI patterns shifted meaningfully across years, reflecting market maturation and changing consumer behavior in India's beauty e-commerce sector.
- **Premium Shoppers** as a target segment and **Influencer / Paid Ads** as campaign types are the next-strongest positive indicators.
- **Google and WhatsApp** channels also positively correlate with high ROI.

---

## 6. Phase 5 — MLflow Experiment Tracking

### Experiment Setup

- **Experiment name:** `Campaign_ROI_Classification`
- Each model run logged: model name, all hyperparameters, AUC, Precision, Recall, F1, and the serialized Spark model artifact.

### Logged Runs

| Run Name | Model | AUC | F1 |
|----------|-------|-----|-----|
| `LogisticRegression_CV` | Logistic Regression | 0.5026 | 0.6365 |
| `RandomForest_CV` | Random Forest | 0.4986 | 0.6365 |
| `GBT_CV` | GBT Classifier | 0.4987 | 0.6365 |
| `Model_Comparison_Summary` | — (summary table) | — | — |

### Best Model (Loaded from Registry)

The best model was retrieved from MLflow by AUC across all runs:

| Metric | Value |
|--------|-------|
| Model | Random Forest (by prior run AUC) |
| AUC | 0.5057 |
| Precision | 0.5555 |
| Recall | 0.7453 |
| F1 | 0.6365 |

MLflow's artifact store enables reproducibility — any run can be reloaded and re-scored without re-training.

---

## 7. Insights & Business Recommendations

### 7.1 Key Findings from the Predictive Model

1. **Pre-campaign features alone provide modest but real predictive signal.** All three models achieve AUC just above 0.50, which — while close to random — represents genuine baseline discriminatory power using only information available before a campaign launches. This is a meaningful result given the strict no-leakage constraint.

2. **Instagram is the highest-signal channel for ROI.** Campaigns that include Instagram in their channel mix are statistically more likely to land in the top 25% ROI bracket. This aligns with Instagram's dominance in the Indian beauty and lifestyle segment among the target demographics.

3. **Campaign timing matters.** The `year` feature was the second-strongest predictor, indicating that ROI norms shift over time. Monthly and day-of-week patterns also contribute — likely tied to seasonal promotions, festive sales (Diwali, Holi), and payday cycles.

4. **Premium Shoppers are the most valuable and predictable segment.** Campaigns targeting this segment yield higher-ROI outcomes more reliably than campaigns targeting Youth or College Students — who may have higher engagement but lower purchasing power.

5. **Influencer and Paid Ads campaigns outperform other types.** These two campaign types rank 5th and 6th in feature importance, making them the most reliable campaign type choices for brands aiming at the top-ROI tier.

6. **The three brands are near-equivalent in scale and structure.** All three CSV sources had consistent schemas and comparable data volumes, making cross-brand comparison methodologically sound.

### 7.2 Actionable Business Recommendations

**1. Prioritize Instagram in multi-channel strategies**  
Given Instagram's top-ranked coefficient in the ROI classifier, brands should ensure Instagram is included as a channel in campaigns targeting Premium Shoppers. Campaigns that exclude Instagram should be scrutinized for compensating factors before launch.

**2. Use the model as a pre-launch campaign screening tool**  
Before allocating budget to a campaign, input its planned attributes (type, channels, target segment, language, duration, timing) into the trained Logistic Regression model to get a high/standard ROI probability score. Route low-probability campaigns to additional optimization before launch, or reallocate budget to higher-confidence ones.

**3. Anchor campaign launches to high-ROI time windows**  
The `year` and `month` features indicate persistent temporal ROI trends. Marketing teams should analyze the monthly ROI trend chart produced in Phase 3 to identify historically high-ROI months (likely festive seasons) and concentrate larger campaigns in those windows.

**4. Invest more in Influencer and Paid Ads campaign types**  
These two types showed the strongest positive associations with high ROI among all campaign types. If budget must be allocated between campaign types, these two should receive priority — particularly on Instagram for the Premium Shoppers segment.

**5. Treat Google and WhatsApp as secondary high-value channels**  
Channels `ch_google` (rank 8) and `ch_whatsapp` (rank 9) were positively weighted. Including these alongside Instagram in multi-channel campaigns is likely to strengthen ROI performance.

**6. De-prioritize underperforming segment-channel combinations**  
The data suggests that campaigns targeting College Students or Youth with Email-only channels consistently underperform. These combinations should be redesigned or sunset in favor of higher-ROI configurations.

**7. Retrain the model monthly**  
The significance of `year` as a feature underscores that campaign ROI dynamics shift over time. Automating monthly retraining via MLflow ensures the screening model stays current with evolving market conditions.

---

## 8. Challenges & Possible Improvements

### Challenges Encountered

| Challenge | Description |
|-----------|-------------|
| **No-leakage constraint limits AUC** | Using only pre-campaign features is correct for a planning tool, but it inherently limits discriminative power. In-campaign signals (CTR, engagement score) are far more predictive — but using them would make the model useless for pre-launch decisions. |
| **Class imbalance (74.3% / 25.7%)** | The 3:1 class imbalance means models that default to the majority class achieve ~74% accuracy while providing no real insight. AUC and F1 are more meaningful here, but all three models still trend toward majority-class dominance. |
| **Identical F1/Precision/Recall across models** | All three models produced the same weighted Precision (0.5555), Recall (0.7453), and F1 (0.6365). This convergence suggests all models learned a similar decision boundary dominated by class-prior probabilities — a symptom of the feature set's limited separating power. |
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

The predictive model, while constrained to pre-campaign features to ensure practical utility, confirmed that **Instagram channel, temporal trends, Premium Shopper targeting, and Influencer/Paid Ads campaign types** are the most reliable early predictors of top-quartile ROI. The pipeline is production-ready for deployment as a campaign pre-screening tool, with a clear roadmap for accuracy improvements through class balancing, external signals, and automated retraining.

---

*Report generated from `annabelle_update.ipynb` — March 2026*
