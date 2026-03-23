# Model card — Credit Card Fraud Detector v1.0

## Model details
- **Type**: XGBoost gradient boosted trees
- **Version**: 1.0.0
- **Trained**: on Kaggle Credit Card Fraud dataset
- **Features**: 34 (V1-V28 PCA + 6 engineered)
- **Decision threshold**: 0.15 (cost-optimised)

## Intended use
- **Primary use**: real-time fraud screening of card transactions
- **Out of scope**: sole decision-maker for account termination,
  credit scoring, or any decision requiring human review

## Performance (test set - 56,962 transactions)

| Metric              | Value   |
|---------------------|---------|
| AUPRC               | 0.882   |
| F1 (fraud class)    | 0.845   |
| Precision           | 90.9%   |
| Recall              | 81.6%   |
| False positives     | ~8      |
| False negatives     | ~18     |
| Latency (p99)       | ~15ms   |

## Limitations & known failure modes

1. **Concept drift**: fraud patterns evolve. Model should be
   retrained every 30-90 days on fresh transaction data.

2. **PCA opacity**: V1–V28 are anonymised — impossible to
   audit which real-world signals drive predictions.

3. **Dataset geography**: training data is from European
   cardholders (2 days, Sept 2013). Performance on other
   regions or time periods is unknown.

4. **Cold start**:  no transaction history means velocity
   features (is_night, is_micro) carry less signal for
   new cardholders.

5. **Adversarial risk**: sophisticated fruadsters who
   understand the model's signals could evade detection
   by mimicking legitimate transaction patterns. 

## Ethical considerations
- False positives cause real customer friction — declined
  legitimate transactions erode trust.
- False negatives cause real financial harm — missed fraud
  hits cardholders and issuers.
- Threshold should be reviewed by domain experts, not set
  purely by automated metric optimisation.
- Model should not be the sole basis for any irreversible
  customer action (e.g. account closure).

## Caveats
Built as a learning project. Not validated for production
use without further testing on live transaction streams,
A/B evaluation, and regulatory review.
