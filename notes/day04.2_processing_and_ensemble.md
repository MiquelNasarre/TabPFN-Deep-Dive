# Day 04.2 — Preprocessing, Postprocessing and Ensemble in TabPFN-2.5

We have looked so far at the entire NN behind TabPFN-2.5 and how it was trained, 
but we are still missing a key component to understand how TabPFN is so successful
with real-world tabular data, that is its data processing.

That part has proven to be a very obscure part of the code, hidden in abstraction,
but thanks to the amazing patience ChatGPT has when reading code and surgical incisions
in the actual `TabPFNRegressor` and `TabPFNClassifier` we are now able to fully describe
the entire pipeline from raw data to processed output.

## Pre-Pre-Processing

Since we aim to cover the entire pipeline it is worth mentioning that previous to the preprocessing steps 
we mention some housekeeping transformations are applied to the input data. These are the following:

- dtype fixing (`fix_dtypes`)
- text/*NaN* handling for *pandas*
- ordinal encoding via `process_text_na_dataframe(...)` + `get_ordinal_encoder()`

This happens during dataset initialization for classifier and regressor. These are not that relevant but 
helps us complete the entire data pipeline.

## Preprocessor types

There is an entire set of classes inside the code dedicated to preprocessing the data 
given as input, then these classes are combined into a stack to be run for each classifier/regressor 
instance, these are all the classes and what they do:

#### `AdaptiveQuantileTransformer`

A safer `sklearn.preprocessing.QuantileTransformer` that automatically reduces `n_quantiles`
based on available samples (and caps it to 20% of subsample) so it doesn’t error or get too 
slow after subsampling. Used for `quantile_*` feature transforms. 

#### `AddFingerprintFeaturesStep`

Adds one extra numeric feature per row: a stable hash “fingerprint” derived from the entire 
row values (salted).This lets the model distinguish duplicate/near-duplicate rows in a 
row-order-invariant way. 

#### `DifferentiableZNormStep`

A torch-based z-normalization: stores per-feature mean/std during fit and applies `(X-mean)/std`.
This is only used if the pipeline is run in differentiable mode (not by default). 

#### `EncodeCategoricalFeaturesStep`

Encodes categorical columns using one of:

* `ordinal` / `ordinal_shuffled`: They are assigned an ordinal value.
* `onehot`: They are one hot encoded into a binary vector.
* `numeric` / `none`: no categorical transformer applied.

#### `KDITransformerWithNaN`

A KDI transform (from `kditransform`) that handles NaNs by:

* imputing NaNs with per-column mean for fitting/transform
* applying the KDI transform
* re-inserting NaNs according to the original NaN mask.

#### `NanHandlingPolynomialFeaturesStep`

Optionally creates random polynomial interaction features:

* standardizes via `StandardScaler(with_mean=False)`
* selects random feature index pairs
* appends products `x_i * x_j` to original features.

Only used if polynomial features are enabled (not by default).

#### `SafePowerTransformer`

A safer `PowerTransformer` replacement that uses a more numerically stable Yeo–Johnson implementation. 

#### `ShuffleFeaturesStep`

Reorders columns after preprocessing:

* `shuffle` (default): random permutation of feature indices
* `rotate`: cyclic shift by a fixed offset
* `None`: no change

#### `SquashingScaler`

A robust scaling + soft clipping transformer (from skrub/RealMLP-inspired):

* Robustly centers/scales per column (RobustScaler / custom minmax / zeros for constant).
* Then applies soft clipping to bound values into `[-B, B]` smoothly.
* Maps ±inf to ±B, preserves NaNs.

#### `RemoveConstantFeaturesStep`

Removes features that are constant on the training data.

* **Fit**: computes a boolean mask `sel_` selecting columns that vary across training rows.
* **Transform**: applies that mask: `X[:, self.sel_]`. 

#### `ReshapeFeatureDistributionsStep`  (Wrapper Class for Transformations)

This is the core numeric preprocessing step, wrapper for other transformer classes. It:

* optionally subsamples 500 features and remembers which features were kept. 
* chooses a named per-feature transformer based on `transform_name`. 
* can optionally append original features alongside transformed ones. 
* optionally inserts a global transformer (like SVD) applied to the whole transformed matrix. 

Transforms include all the other discussed classes 
plus many KDI variants if available (`kdi_alpha_*`).

## Classifier Default Preprocessing

Both classifier and regressor have two different preprocessing paths by default, meaning
the forward passes of the ensembles will use one or the other. These are the 
two preprocessors for the default `TabPFNClassifier`:

#### Estimator A (squashing + ordinal cats + SVD + fingerprint + shuffle)

* `ReshapeFeatureDistributionsStep('squashing_scaler_default' + 'svd_quarter_components', ...)`
* `EncodeCategoricalFeaturesStep(categorical_transform_name='ordinal_very_common_categories_shuffled')` 
does ordinal encoding with a heuristic that only treats sufficiently common low-cardinality columns as 
categorical, and it shuffles category IDs for robustness.
* `AddFingerprintFeaturesStep` 
* `ShuffleFeaturesStep(shuffle_method='shuffle', ...)` 

#### **Estimator B (“none” + numeric cats + fingerprint + shuffle)**

* `ReshapeFeatureDistributionsStep(transform_name='none', ...)`
* `EncodeCategoricalFeaturesStep(categorical_transform_name='numeric')`
* `AddFingerprintFeaturesStep` 
* `ShuffleFeaturesStep(shuffle_method='shuffle', ...)` 

There is basically one path that applies some squashing, globally SVD, and encodes categorical 
features while the other path mainly leaves the features untouched, both do add fingerprint columns
and shuffle the features.

## Regressor Default Preprocessing

These are the two preprocessors for the default `TabPFNRegressor`:

#### Estimator A (quantile-to-uniform + maybe append-original + numeric cats + fingerprint + shuffle)

* `ReshapeFeatureDistributionsStep(transform_name='quantile_uni_coarse', append_to_original='auto', ...)`, 
appends the original features if the count is below a certain threshold.
* `EncodeCategoricalFeaturesStep('numeric')` (no categorical transform) 
* `AddFingerprintFeaturesStep` 
* `ShuffleFeaturesStep('shuffle')` 

#### * B (squashing + ordinal cats + SVD + fingerprint + shuffle)

* `ReshapeFeatureDistributionsStep('squashing_scaler_default' + 'svd_quarter_components')` 
* `EncodeCategoricalFeaturesStep('ordinal_very_common_categories_shuffled')` 
* `AddFingerprintFeaturesStep` 
* `ShuffleFeaturesStep('shuffle')` 

Additionally, the regressor config includes:

* `REGRESSION_Y_PREPROCESS_TRANSFORMS = (None, 'safepower')`

This means the regressor ensemble also tries a target transform of `y`, using `SafePowerTransformer`. 

In essence one path does a quantile transformation and the other path does squashing and SVD and then 
encodes categorical features, both paths add a fingerprint feature and shuffle. And additionally in the 
regressor a target power transformation can be applied.

After preprocessing, the processed $X_{train}/X_{test}$ and processed targets are packed into the PFN 
prompt format (support + query tokens, label injection/masking), then fed through the Transformer, which links
us to yesterday's report.

## Post-processing + Ensemble

Both the classifier and regressor run 8 forward passes by default, 4 for each preprocessor with different
random seeding. After the forward pass we are left with the raw logits, now we will review what happens 
with those logits to get converted to the actual output.

#### Classifier

First, temperature scaling is applied to the logits (0.9 by default). After that there are two options, 
average before softmax, or after (default), which averages the output before or after applying softmax. 

Then, optionally, class balancing is applied, which reweights the predicted distribution based on the 
training class counts. Finally, probabilities are returned.

#### Regressor

First, temperature scaling is applied to the logits (0.9 by default).

After that, since some target transformations may have been applied, the buckets borders need to match 
in all settings so they are adjusted and logits interpolated so that all correspond to the same buckets.

For each estimator, they adjust borders according to y-transform and cancel invalid bins.
Then they translate each estimator’s distribution onto the original bucket borders.

Ensemble-average translated distributions (with the same two modes as the classifier) and convert to 
log-probabilities. Finally map log-prob distribution to mean/median/mode/quantiles as specified by the user.

## Customization & fine-tuning

So what can we control from TabPFN-2.5 as an end user, without the use of any extensions of retraining?
There are some basic tweaks we can use, all can be found in the class constructor, and they include 
temperature control, number of estimators, `average_before_softmax`, `differentiable_input`, provide the 
list of categorical features for easier inference via `categorical_features_indices` and some advanced 
settings to be found inside `InferenceConfig`. 

There are other efficiency related options I am ignoring for the moment. These are the basic tweaks the 
model provides us for fine-tuning. Quite limited given the complexity of the forward pass.

## Conclusions

We now fully understand the entire pipeline of the data from raw input to probabilities inside TabPFN-2.5.
We can definitely see that it is a complex system and now we will be able to analyze the claims made in 
their papers with a much deeper foundation.

Now it is also time to get familiar with the TabPFN-2.5 API itself, learn all its capabilities and ecosystem
outside the underlying models we've been focusing on since we started this deep dive. With this acquired 
knowledge we will clearly be able to make better decisions when fine-tuning our models, and understand what 
is exactly going on underneath.

It is also time to start thinking about market data, for example looking at the time-series expansion and 
considering its potential for market forecasting, and seeing which tuning we need to do and how we should 
use the output distributions to make useful predictions and portfolio decisions.

I am looking forward to this new exploration and experimentation phase.

## Open questions

- What TabPFN extensions can be useful when analyzing different types of datasets?
- What other knobs and capabilities does TabPFN offer besides the basic ones we discussed today?
- How will TabPFN perform when faced with noisy market data?
- What limitations will we find when experimenting with TabPFN-2.5?
- Can I replicate a small PFN pretrained NN for testing given my compute limitations?
