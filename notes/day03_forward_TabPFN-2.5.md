# Day 03 — Down the forward pass of TabPFN-2.5

Today was supposed to be a day of analyzing TabPFN-2.5 papers and technical reports.
But, after reading its technical report I felt like I could not really grasp their 
claims without actually understanding how TabPFN-2.5 works. 

Thus the days of exploring the rabbit hole of TabPFN code begin. Most papers are not 
specific at all when talking about the architecture of the NN, and then there is me, 
who likes to get his hands dirty in any code it touches, so I decided to organize the 
following days as follows. 

Today I looked at how the forward pass of TabPFN works, both for classification and 
regression, from preprocessed input data all the way to output. Avoiding the lengthy 
preprocessing, that will be covered soon.

Tomorrow I will look at the training regime, especially the **prior** generation, 
which is arguably the most important part of a PFN as seen yesterday.

The next day we will analyze the preprocessing magic, as well as the ensemble and 
customization/fine-tuning features TabPFN-2.5 provides.

Let's get started.

## Humble beginnings (Prepare for encoder)

Let us start with preprocessed tensors $X_\text{train}$, $y_\text{train}$ and 
$X_\text{test}$, of shapes $\( B, S_\text{train}, F\)$, $\( B, S_\text{train}\)$ and 
$\( B, S_\text{test}, F\)$ respectively. Where $B$ is the batch size, $S$ are the number
of samples in the train and test set ($S = S_\text{train} + S_\text{test}$), and $F$ is 
the number of features.

To actually make the data prepared for the encoder we first concatenate the train and 
test sets, giving us $X$ of shape $\( B, S, F\)$ and $y$ of shape $\(B, S\)$, for the 
test rows of $y$, we set the values to *NaN* to make sure there is no leakage and so 
that the encoder can identify it as missing values.

Then we will split the features on feature groups of 3 features each padding the necessary
features so that the total number is divisible by 3 ($3|F+k$), obtaining then the number
of feature groups $F_g = \(F+k\)/3$ and reshaping $X$ into $\( B, S, F_g, 3\)$.

**NOTE:** In the actual code the first two dimensions of the tensors are transposed in 
multiple places obtaining tensors of the following shape $\( S, B, ...\)$. For clarity, 
since this poses no functional difference to our explanation we will simply ignore it.

## X Encoder, Y Encoder, Embedding

The actual encoder used by TabPFN is nowhere to be found in the code, and is not specified 
in any of the papers I have read so far. Luckly we have access to the model, so we can 
disect it and extract it. In `basic_tests.ipynb`, I obtained the following encoders from both 
X and y for the Classifier and the Regressor

```
Encoder: SequentialEncoder(
  (0): RemoveEmptyFeaturesEncoderStep()
  (1): NanHandlingEncoderStep()
  (2): VariableNumFeaturesEncoderStep()
  (3): InputNormalizationEncoderStep()
  (4): VariableNumFeaturesEncoderStep()
  (5): LinearInputEncoderStep(
    (layer): Linear(in_features=6, out_features=192, bias=False)
  )
)
Y encoder: SequentialEncoder(
  (0): NanHandlingEncoderStep()
  (1): MulticlassClassificationTargetEncoder()
  (2): LinearInputEncoderStep(
    (layer): Linear(in_features=2, out_features=192, bias=True)
  )
)

Encoder: SequentialEncoder(
  (0): RemoveEmptyFeaturesEncoderStep()
  (1): NanHandlingEncoderStep()
  (2): VariableNumFeaturesEncoderStep()
  (3): InputNormalizationEncoderStep()
  (4): VariableNumFeaturesEncoderStep()
  (5): MLPInputEncoderStep(
    (mlp): Sequential(
      (0): Linear(in_features=6, out_features=1024, bias=False)
      (1): GELU(approximate='none')
      (2): Linear(in_features=1024, out_features=192, bias=False)
    )
  )
)
Y encoder: SequentialEncoder(
  (0): NanHandlingEncoderStep()
  (1): LinearInputEncoderStep(
    (layer): Linear(in_features=2, out_features=192, bias=True)
  )
)
```

Let's break down what each of those modules does to get a better picture:

- `RemoveEmptyFeaturesEncoderStep`: Detects constant features on the training rows and 
removes them. For batch size 1 it drops them, for batch size > 1 it packs non-empty 
features to the front and zero-pads the remainder to keep a fixed width.

- `NanHandlingEncoderStep`: If it finds *NaN*s it replaces them by the average value on 
the training set and flags them as *unknown*, creates an additional flag channel with the 
same width as the values channel; later the channels are concatenated, turning $X$ into 
$\( B, S, F_g, 6\)$ and $y$ into $\( B, S, 2\)$.

- `VariableNumFeaturesEncoderStep`: Transforms the input to a fixed number of features by 
appending zeros to the removed features. Also normalizes the input by the number of used 
features to keep the variance of the input constant, even when zeros are appended.

- `InputNormalizationEncoderStep`: Encoder step to normalize the input in different ways. 
Can be used to normalize the input to a ranking, remove outliers, or normalize the input 
to have unit variance.

- `MulticlassClassificationTargetEncoder`: Orders the labels from smaller to biggest 
assigning them a value from 0 to total number of labels. This value is then directly fed 
into the linear layer.

- `LinearInputEncoderStep`/`MLPInputEncoderStep`: Encoders that perform the linear/MLP 
projection of the input data.

$X$ gets transformed into $\( B, S, F_g, 6\)$ with the additional information of the flags.
Then for the Classifier it simply gets linearly transformed into the embedding dimension
$\(6\rightarrow 192\)$, obtaining $\( B, S, F_g, 192\)$. For the regressor an MLP is used 
instead consisiting of 
$\(6\rightarrow 1024\) \rightarrow \text{GeLU} \rightarrow \(1024\rightarrow 192\)$, 
obtaining the same output shape but allowing for a richer initial embedding.

$y$ in both cases gets transformed into $\( B, S, 2\)$ to encode the missing labels flag 
for the test set, and are linearly projected into the embedding $\(2\rightarrow 192\)$, 
with the Classifier having the additional target encoder to nicely order the labels.

After encoding, TabPFN adds a feature-group identity embedding to $X$. It samples a seeded 
random vector in a low-dimensional subspace $\(E/4=48\)$, projects it to $E=192$, and adds 
it to each feature group. This breaks strict feature-order invariance but gives the model 
stable feature identities.

To obtain the combined embedding we unsqueeze the $y$ vector to $\( B, S, 1, 192\)$ and 
concatenate it with $X$ obtaining $emb = cat(X,y)$ of shape $\( B, S, F_g+1, 192\)$.

Before sending this tensor to the transformer we do one last step, we add $64$ thinking 
rows, these are learned rows tokens at the begining of the tensor that the transformer 
can use for storing information about the set, and have proved to be very valuable on 
small datasets. Therefore the final $emb$ tensor has the shape 
$\( B, 64+S, F_g+1, 192\)$.

## Transformer (Double Attention)

When you have tabular data you don't have just a list of tokens, instead you have an 
array of them, and you want them to communicate all with each other independently of how 
the rows and columns are arranged in your data, therefore you need to adapt the attention
architecture to this environment.

The way TabPFN does it is quite clever, simply put, we first attend the features, and 
then attend the rows. To do that we first arrange $enc$ as $\( B * \(64+S\), F_g+1, 192\)$,
and we run attention between the feature groups of each individual row, simple $3$-headed 
self-attention, we add residual and layer normalize. Then we transpose $enc$ to 
$\( B * \(F_g+1\), 64+S, 192\)$, and we run attention again, this time the attention will 
run between the same feature groups of all the rows, here we will apply an attention mask 
so that training rows attend all to each other and test rows only attend to training rows.
Then again we add residual and layer normalize.

To finish off we reshape $enc$ back to normal $\( B, 64+S, F_g+1, 192\)$ and we pass the 
tensor to a feed forward layer with a hidden dimension of $1024$, again we add residual 
and layer normalize.

This process is repeated for $18$ layers for the Regressor and $24$ layers for the 
Classifier. Before obtaining the transformer output $out$ of shape $\( B, 64+S, F_g+1, 192\)$.

## Decoder and post-processing

### Classifier

The decoder for the Classifier is quite straightforward, we take only the last feature 
(originally the encoded target) from all the test rows obtaining a tensor of shape 
$\( B, S_\text{test}, 192\)$, and we pass it through an MLP with a hidden layer of size
$1024$, $\(192\rightarrow 1024\) \rightarrow \text{GeLU} \rightarrow \(1024\rightarrow 10\)$, 
to obtain the $10$ raw logits, then some post-processing steps are done we might further 
discuss another day, mainly clamping to the actual number of labels you are working with, 
adding temperature and applying softmax to turn the logits into probabilities.

## Regressor

For the regressor the story is a bit more interesting. TabPFN proudly announces that its 
Regressor does not output a number but instead it outputs a probabilty distribution. The 
way they do that is by discretizing the real number line into $5000$ buckets, and 
outputting a probability for each one of those buckets.

We also start by taking only the last feature (originally the encoded target) from all the 
test rows obtaining a tensor of shape $\( B, S_\text{test}, 192\)$, and we pass it through 
an MLP with a hidden layer of size $1024$, 
$\(192\rightarrow 1024\) \rightarrow \text{GeLU} \rightarrow \(1024\rightarrow 5000\)$, to 
obtain the $5000$ raw logits.

Each one of those logits map to a bucket, the bucket boundaries are decided by a preprogrammed 
initial distribution scaled by the $y_\text{train}$ mean and std. We can now apply some post
processing and softmax to obtain the final probability distribution, which can be used to 
compute the mean, variance, median, etc, or to study the distribution itself.

This output really follows the spirit of the Bayesian foundations of the model, and gives it 
a remarkable difference from most of the competition.

## Conclusions

TabPFN makes a lot of claims about its performance, but to truly understand it a deep dive 
is clearly necessary, only the forward pass of the NN is much more complex than the entirety 
of its predecessor architecture, and from this architecture some very nice properties arise, 
like almost column permutation invariance, a fully connected tabular analysis and the 
capability of outputing distributions instead of single predictions.

It is true though that column permutation invariance is broken by the feature groups and even 
further by the positional embeddings, however, this is probably mitigated by the training 
regime, topic we will further explore tomorrow.

I am excited to follow tomorrow with the training process, to further understand how TabPFN 
really works and maybe to be able to build my own in the near future.

## Open questions

- How was TabPFN-2.5 prior generated and what was the training regime?
- How does the TabPFN preprocessing and post-processing work?
- How does the previous question tie with ensemble and fine-tuning of the model?
- Can I replicate a small PFN pretrained NN for testing given my compute limitations?
- How will TabPFN perform when faced with noisy market data?