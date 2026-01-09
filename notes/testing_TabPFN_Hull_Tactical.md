# Notes on testing TabPFN-2.5 on the Hull Tactical competition dataset

Yesterday I ran some tests on TabPFN-2.5 with the Hull Tactical data, and before I lose my thoughts I will
write them down in these notes. 

Concretely, I ran the tests using both the raw data and my preprocessed data and compared the results to the 
ensemble model I submitted in the competition, a $48$ parameter model with a $200$ bootstraps trained ensemble.

## Methodology and criticisms

The models were trained in the `date_id` range from $1000$ to $5000$ and tested on the range from $5000$ to 
$7000$. Not starting from zero is due to the lack of most of the features during the first $1000$ days. This 
is a considerable downsampling from the total amount of data, which can clearly affect performance. This 
choice is made because I am currently developing a PFN that will also be tested, and fine tuning will also 
be done on these conditions. The final evaluation will be trained once on the range $1000$ to $7000$ with the 
chosen hyperparameters and tested once on the range $7000$ to $9000$. This will ensure all models are tested 
on an unbiased setting. Of course, my ensemble is biased because the preprocessed data was chosen to suit it 
in the entire dataset, but it is what it is.

The output of the models is to be an S&P500 position for each test market date, the position must be between
$0$ and $2$ ($0$% to $200$%). Later we will cover the specifics of how to transform output to positions.

For the metrics I am using two different ones. The first one is the modified Sharpe ratio introduced by the 
Hull Tactical competition. I have a function that exactly mimics that ratio, it takes as input the model 
positions, the forward returns, and the risk-free rate and outputs a real value similar to a Sharpe ratio.
The second one is strategy returns, though not as important since it is not part of the competition metric 
that I specifically trained my ensemble for, it is always nice to see profits, so if bigger profits can be 
obtained without compromising much the Sharpe ratio metric that is also considered positive. Daily profits 
are computed via the formula

$$
\text{log profits} = \text{log returns} · \text{position} + \text{log risk free} · (1 - \text{position}),
$$

then cumulatively summed to obtain strategy profits on the test set.

Now how do we transform predictions to positions: We take the raw predictions of the model, we multiply them 
by a scaling factor and move them by a bias and we clamp them to the interval $[0,2]$. The scaling factor and 
bias for this test set have been chosen to maximize the Sharpe ratio. Since the best scale and bias for TabPFN 
were still too conservative, they achieved good Sharpe ratios but almost no profits, so as mentioned before I 
forced a bigger scale to increase the profits while slightly affecting the Sharpe ratio. On the final test only 
scales and biases previously distilled from this initial test will be used, to make it completely unbiased.

### Strong criticisms

There is absolutely no doubt that the methods are heavily biased towards my ensemble model and that TabPFN 
is not used to its full capabilities at all, there is one simple reason for that, time constraints. 

* Using mean output instead of the full distribution provided by the PFN and the same method I used to transform 
predictions to positions for my ensemble clearly affects the capabilities of TabPFN-2.5. 

* Using the preprocessed features I designed for my ensemble for TabPFN is not ideal, instead, its preprocessing 
should be tailored specifically for TabPFN, still as you will see performance does improve.

* Using the Sharpe ratio metric for which my ensemble was trained for gives an unfair disadvantage to TabPFN.

* No fine tuning is done on the TabPFN-2.5 architecture, this is due to fine tuning appearing limited for 
TabPFN-2.5, only having the option of a bigger ensemble or a different temperature on the user hyperparameters.
Although deeper fine tuning is most certainly possible, the TabPFN repository only mentions specialized features 
as a way of fine tuning, which is provided via our preprocessing, so no further attempts to fine tune it have 
been done.

## Results

Before analyzing the results it must be stated that the test period is quite a tricky one for trading, since it 
is mostly a slow constant growth period, which by itself it has a big Sharpe ratio and not too much margin to 
obtain profit.

Despite all the limitations mentioned, TabPFN-2.5 has proven to be useful for analyzing market data. The models 
compared are: constant position 1, constant position 2, randomized predictions, my ensemble, TabPFN-2.5 with 
raw data, TabPFN-2.5 with my preprocessed data. These are the results:

| Model                              | Sharpe Ratio | Returns   |
|------------------------------------|--------------|-----------|
| Constant Position (1x)             | 0.941        | 1.097     |
| Constant Position (2x)             | 0.482        | 2.180     |
| Randomized Predictions (SR max)    | 0.4 – 0.8    | 0.4 – 1.4 |
| My Ensemble (SR max)               | 1.134        | 1.590     |
| TabPFN-2.5 Raw (SR max)            | 1.008        | 1.072     |
| TabPFN-2.5 Raw (Riskier)           | 0.939        | 1.363     |
| TabPFN-2.5 Preprocessed (SR max)   | 1.101        | 1.043     |
| TabPFN-2.5 Preprocessed (Riskier)  | 1.052        | 1.526     |

## Conclusions

*TabPFN-2.5 and PFNs in general can be used to analyze market data.*

*You should not skip human data analysis for market data.*

Those are the two strongest conclusions I can take from these preliminary tests, as can be seen from the results
TabPFN will always outperform randomized predictions, which is a non-trivial benchmark for market data, and if 
fed preprocessed data — even if it is not taylored to it — it will clearly improve its predictions.

Since these results are biased due to the maximization of Sharpe ratio given the target data, and also the test
set is quite small, not too much can be stated of the results other than what has already been said.

Now to my gut instincts, given the clear limitations of these tests and the fact that we still get positive 
results out of them shows that the PFNs are most likely very capable of market data analysis. If the PFN was 
specifically trained on market-like data, the preprocessing was taylored to the PFN capabilities and it was 
fine tuned for our specific dataset the PFN would almost surely outperform a simple MLP ensemble like mine in 
all metrics. Now time to develop it!

## Notes on TabPFN-TS

I find important to state why TabPFN-TS was not included in these preliminary tests, and what can we learn from
it despite its exclusion. 

The main limitation of TabPFN-TS is that it only allows for a single feature, the lagged target. This is because 
the way it works is by applying different transformations to the time sequence and encoding the obtained values 
as features. 

For example it encodes day of month, hour of day, minute of hour and second of minute as sine and cosine waves 
and adds them to the feature list. It also performs a fourier transform to extract most relevant frequencies and 
adds them as well.

Then the original lagged target and the new features are directly fed into the TabPFNv2 Regressor and are fitted 
for later predict calls.

This simple idea obtains competitive results on time series analysis and outperforms most of the state-of-the-art 
models, so, what can we learn from it? Since market data is also time dependent it would be a smart choice to 
encode time features into our feature list so that when the data is fitted to the PFN this has a *"sense of time"* 
from seeing those features. Sine and cosine functions seem to work really well for encoding time.

The TabPFN-TS was not included since it would only use the lagged target as input, and given the noisiness and 
the lack of other data it will probably not perform very well, but I might be wrong. Also further tests could 
be done including time features, although some are present in my preprocessed data.
