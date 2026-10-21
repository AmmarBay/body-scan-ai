# NHANES DXA Body-Fat Dataset

This project can build a model-ready dataset from the CDC/NCHS NHANES
1999-2000, 2001-2002, 2003-2004, and 2005-2006 public-use files.

The target is `DXDTOPF` (DXA total percent fat). Input measurements are age,
sex, weight, standing height, and waist circumference, with BMI derived from
weight and height. The preparation script keeps adults ages 18-69 and retains
the five NHANES DXA imputation versions as separate rows per participant.

## Prepare and train

From the repository root:

```bash
python prepare_nhanes_dxa.py
python train_nhanes_model.py
```

The preparation script downloads the required XPT files from the official
[CDC NHANES public data archive](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx)
into the ignored `raw/` folder and writes `nhanes_dxa_bodyfat.csv` here. The
trainer evaluates each imputation separately and splits by participant to
prevent the same person's imputation records from appearing in both train and
test sets. Its output is `bodyfat_nhanes_dxa_model.pkl`, which the app uses
for adult estimates.

## Important interpretation

This source has a different reference target from the existing BodyFat
dataset: DXA total percent fat is not interchangeable with a body-fat estimate
derived from another measurement method. Do not compare R-squared values across
those targets as if they were the same task.

NHANES DXA data have five multiple-imputation versions. The CDC recommends
using all five versions for statistical analyses, not selecting one version or
averaging their values. The trainer therefore fits and evaluates
one model per imputation version and reports variation across them. The app
averages the five model predictions; it does not average the imputed target
values. The published metrics are predictive validation scores, not
population estimates.

NHANES represents the United States and the measurements were collected in
1999-2006. It may not represent current users or the Indonesian population.
The model is restricted to ages 18-69 and must not be treated as a
medical diagnosis.

## Source documentation

- [Whole-body DXA data and multiple-imputation documentation](https://wwwn.cdc.gov/nchs/nhanes/dxa/dxa.aspx)
- [NHANES DXA FAQ](https://wwwn.cdc.gov/nchs/nhanes/dxa/dxafaq.aspx)
- [NHANES 2005-2006 DXA codebook](https://wwwn.cdc.gov/nchs/data/nhanes/public/2005/datafiles/dxx_d.htm)
