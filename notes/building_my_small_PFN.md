# Notes on `my_small_PFN.py`

This one will not be as extensive as the other notes since this is mostly an 
implementation and there is not much magic to it.

I just finished building my first PFN and I want to go over the exact architecture
and how it is inspired by TabPFN-2.5, the preliminary tests done to make sure it is 
working okay, and offer some thoughts on the matter.

## Overall structure mimicking TabPFN-2.5

First let's break down the structure into its different modules and explain what 
each module does:

#### Preprocessing

Final Shapes: $X \rightarrow (B, S = train_size + test_size, F)$, 
$y \rightarrow (B, train_size, 1)$

We start by the preprocessing of the data. This in TabPFN is a resource heavy step,
but since we aim to simplify the overall flow we assume that the data has already been
analyzed, processed and normalized by the user. This step then is mostly about shape 
matching and preparing the $X$ and $y$ tensors for the encoders.

#### Encoder X

Final Shapes: $X_emb \rightarrow (B, S, Fg, E)$

This step splits the tensor features into feature groups, padding with empty features 
and scaling if necessary to make sure all groups contain the same amount of features, 
then applies the encoding to output the embedded tokens.

As a reminder in the TabPFN architecture each feature group of each row is tokenized,
so instead of holding a vector of tokens per batch, we hold a matrix of tokens per batch.

#### Encoder y

Final Shapes: $y_emb \rightarrow (B, S, 1, E)$

This step appends the test rows as zeroed tensors, creates the mask to encode them as 
empty and concatenates it to the tensor, and then applies the encoding to output the 
embedded target tokens, reshaping them to match the embedded $X$ format.

#### Concatenate and add thinking rows

Final Shapes: $trans_in \rightarrow (B, S+T, Fg+1, E)$

To create the transformer input the embedded $X$ and $y$ tensors are concatenated along
the feature dimension, and then the thinking rows are prepended. Each row has a learned 
token that is broadcasted across all the features, matching the architecture used by 
TabPFN for the thinking tokens.

#### Transformer/Layer/Double Attention/Feed Forward

Final Shapes: $trans_out \rightarrow (B, S+T, Fg+1, E)$

Since the transformer architecture has already been covered in detail I will keep the 
explanation brief. But basically it performs attention first between the different feature 
groups in each row (across $Fg+1$), and then it performs attention between the different 
rows for each feature group (across $S+T$). This is done simply by transposing and reshaping 
the embedded tensor.

Some normalizations are applied and a feed-forward step is added at the end of every layer.
So every layer does: feature attention, row attention, feed forward.

#### Decoder

Final Shapes: $logits \rightarrow (B, test_size, n_buckets)$

Mimicking TabPFN-2.5 Regressor the output is a probability distribution for the target 
given by a discretization of the real number line in buckets.

The decoder will take the transformer output and extract the target tokens from the test
rows, then it will apply an MLP to obtain the output logits directly from the tokens.

#### Post-processing

After the logits are obtained and temperature is applied the `predict()` function allows 
for different output types, all these are based on the bucket generation. 

Different functions to operate with the buckets are defined inside the class `BucketOps`. 
This allows to easily convert from bucket probabilities to real probability distributions 
and extract values like the mean or standard deviation.

Since no ensemble is used and simplicity is the objective almost no post-processing steps 
are applied to the data, which differs from the TabPFN approach, but simplifies the model 
architecture.

## Sanity Checks

Different basic tests can be found on `my_PFN_tests.ipynb`. These include: model loading, 
random data generation, converting test targets into labels, performing a forward pass 
performing a backward pass, and checking that the input and output distributions behave 
as expected.

So far all test conducted have turned exactly as expected, so the model is behaving and 
probably ready for training.

## Conclusions

I am quite happy with the architecture so far. The bucket discretization is really cool 
and seems to perform exactly as expected. The model forward pass and backward pass fit 
inside my GPU with my desired data dimensions and so far I am impressed by the speed I 
can get with my computer.

Tomorrow finally I will be able to design the **prior** generation for the model and if 
all goes well I will start the preliminary training tomorrow. I am quite excited to follow
those steps and see if we can make this model learn something. :)
