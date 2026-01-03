# Day 02 — So... What is a PFN anyway?

As the title suggests day 2 was mostly about research. Throughout the day 
I read 4 papers on the topic and I will try to encapsulate the knowledge I
gained in this daily report.

## The core idea behind PFN

> *We assume a given representative prior distribution over supervised learning 
> tasks (or functions), which provides our inductive bias.*

I really like this phrase found in the Nature paper because it clearly encapsulates the main 
idea behing Prior-Data Fitted Networks. You have a function you want to model given by some 
data, and you make an assumption of what type of function that might be to create a Neural 
Network capable of representing it. That is what everyone does when faced with a ML problem, 
but a lot of times the data you have is a very small dataset, and making a model that does not 
overfit and is capable of generalizing that data is usually a slow and painful process.

PFNs aim to solve that problem, by making a general assuption of the kind of function you want 
to represent — For the particular case of TabPFN this is a Structural Causal Model with noise 
and a given limitation on parameters and dataset size — you can train a specialized deep NN on 
a large number of similar synthetic problems, so that it will be able to take your dataset data 
and generalize it with no fine-tuning or prior knowledge, basically performing in context 
learning (ICL).

## More technical overview

> *Tabular data has long been abandoned by deep learning research. ... We propose a radical 
> change to how classification is done.*

We want to create a NN architecture that will take full datasets as input, and will give 
you a model you can use to make predictions on that dataset during inference.

For that we will create a deep NN and train it on synthetic datasets with similar 
characteristics to the ones we want to model IRL. Those pretraining datasets are 
called the **prior**, and will define how our NN designs its model during inference.

Our NN will expect as input a dataset $D_{train} = \{ X_{train}, y_{train} \}$ and the set 
of inputs we want to make inference on $X_{test}$ and will output a probability distribution
for $y_{test}$ given the entire input $p(y_{test}|X_{test}, D_{train})$.

As explained in the 2023 TabPFN paper:

The posterior predictive distribution for a test sample $X_{test}$ specifies the distribution
of its label $p(·|X_{test},D_{train})$, which is conditioned on the set of training samples
$D_{train} := \{(x_1,y_1),...,(x_n,y_n)\}$. 

The PPD can be obtained by integration over the space of hypothesis $\Phi$, where the weight 
of a hypothesis $\phi\in\Phi$ is determined by its **prior** probability $p(\phi)$ and the 
likelihood $p(D|\phi)$ of the data $D$ given $\phi$:

$$
p(y|x,D) \propto \int_{\Phi} p(y|x,\phi)p(D|\phi)p(\phi)d\phi
$$

## PFN general training algorithm

Given what we have seen so far, to train a PFN the ingredients we need are: a NN architecture 
that can take the inputs as specified and output as specified, and a dataset generator based 
on our desired prior distribution. The training algorithm is described as follows:

**Input:** A prior distribution over datasets $p(D)$, from which samples can be drawn and the 
number of samples $K$ to draw.

**Output:** A model $q_\theta$ that will approximate the posterior predictive distribution.

**Training:**
for $j\in\{1,...,K\}$

- Sample $D = D_{train}\cup D_{test} \sim p(D)$ with $D_{test} = \{ (x_i,y_i) \}_{i=1}^m$

- Compute stochastic loss approximation 
$\overline{\ell}_\theta = \sum_{i=1}^m (-\log q_\theta(y_i|x_i,D))$

- Update $\theta$ with stochastic gradient descent on $\Lambda_\theta\overline{\ell}_\theta$

After enough training if your NN architecture is good it will evetually learn to mimic the 
**prior** distribution given new datasets as input, and if the dataset follows a similar 
distribution it will be able to make accurate out-of-sample predictions on it, basically 
performing ICL. We will later discuss how this ICL is even possible.

## Why does it work? (in theory)

The PFN development comes from a purely Bayesian point of view of statistics. This tells us that 
there are some **prior** beliefs we hold on a certain probability distribution, lets call those 
beliefs the **prior** simbolized by the probability measure $\Pi$. We will also consider 
the support $\mathcal{P} = \{p : \Pi(p) > 0\}$ the set of distributions possible in our **prior**.
When those beliefs are met with data, let that be a dataset $D$, we adjust our beliefs on the 
distribution. We call $\Pi(· | D) = P(· | D, \Pi)$ for any $p\in\mathcal{P}$ the posterior, it 
is the conditional distribution of $p$ given the data $D$ and our **prior** $\Pi$. 

PFNs then focus on Posterior Predictive Distributions (PPDs), that aim to find the most likely 
probability distribution inside a dataset based on the **prior** and dataset provided. Let 
$\pi$ be a function induced by the **prior** $\Pi$, that generates PPDs by 
$\pi(·|x,D): Y \rightarrow [0,1]$ for a given dataset $D$ and particular case $x$. This function 
will take a label $y\in Y$ as an input and will output the probability of $x$ having that label
given the **prior** belief, the dataset provided $D$ and the particular case $x$, it has the 
following form:

$$
\pi(y|x,D) = \int_{p\in\mathcal{P}} p(y|x) d\Pi(p|D)
$$

This shows that PPDs are fully characterized by the **prior**, being simply the posterior mean
over conditional distributions $p(y|x)$.

Essentially what we want is a way of approximating $\pi$, with this background is where PFNs
appear. PFNs rely on the fact that under log loss, the Bayes-optimal predictor is the PPD.
Then given a parameterized function $q_\theta$ we want to use to approximate $\pi$, our training
objective becomes 

$$
\min_\theta ​\mathbb{E}_{D\sim p(D)}​\mathbb{​E}_{(x,y)\sim D}​[−\log q_\theta​(y|x,D_\text{train}​)]
$$

This is equivalent to minimizing negative log likelihood, and can be done via SGD. Then if $q_\theta$
is expressive enough, and the optimization succeeds we will have an approximator of $\pi$, 

$$
q_\theta​(·|x,D)\rightarrow \pi(·|x,D)
$$

> If $D_n$ is a data set generated from $p_0$, we hope that $\Pi(p | x, D_n)$ concentrates
> around $p_0$ as the size of $D_n$ increases. Setting a good prior is tricky in a nonparametric 
> context. Finding a prior supporting a large enough subset of possible functions isn’t trivial. And
> even if, the prior may wash out very slowly or not at all if it puts too much mass in unfavorable 
> regions. But also if $p_0$ is outside the support $\mathcal{P}$ of $\Pi$, PPDs can
> learn from data if the prior is sufficiently well-behaved.

I find this section to be important in the moment to create a mindset of what PFNs are, I would 
not consider them a NN in the common sense of trying to approximate a simple function, and I 
will also not consider them some magical ICL machine that learns from the data provided.
Instead I would consider them —in a Bayesian mindset— NNs that try to approximate a PPD, meaning
that they will mimic the prior distribution on new data provided, and produce prediction models
based on that.

You can also think it on a more poetical mindset, where the training Dataset does not make the
NN perform ICL, instead it becomes part of the PFN and its parameters, being the missing piece 
it needed to make predictions tailored to that dataset, and that is not magical, it is by design.

## The TabPFN approach

So far we have outlined the theoretical fundamentals to create and train a PFN, while avoiding 
proofs about how ICL emerges and viability objections on the matter.

Now I want to focus on TabPFN and how they approached their PFN architecture and pretraining, 
first we will see how they designed their NN and then we will focus on their pretraining and 
their **prior** generation.

For today we will simply follow the 2023 paper on TabPFN, we will study TabPFN-2.x further down
the line, so some of the information and limitations that will be given might not apply on the 
newer versions.

ICLR 2023 TabPFN paper: 
<https://arxiv.org/abs/2207.01848>

Source code (as referenced in ICLR 2023 paper):
<https://github.com/automl/TabPFN> (redirects to <https://github.com/PriorLabs/TabPFN>)

## Build your own TabPFN

> *Attention is all you need*

Although this might look like the most mysterious part of TabPFN and where the novel ideas are,
this could not be further from the truth, the model is a simple 12 layer transformer with not 
that much to it. Lets follow the data from input to output to see what it actually does 
(or it did in 2023).

**WARNING**: Some of this information is not directly stated on the paper and the original 
code has been lost in a deep sea of commits with no signs of life. So some of the specifics 
are checked by ChatGPT and therefore cannot be fully trusted (the y injection is a bit weird 
NGL). For the TabPFN-2.5 code exploration information will be more accurate.

You enter your training set with its —up to 100— features and with its label —from 0 up to 9— 
for each training example. Each feature is then normalized and other preprocessings might be 
applied to it like logarithmic scaling if the feature has exponential behaviour, etc. These 
preprocessings have most likely changed for newer versions so they are not that important.

If features are less than 100 the remaining rows are zero padded and the features are scaled to 
maintain overall density. The target label appears to be treated as a real value, normalized and 
directly injected into the input vector as is, with an additional last column to indicate that 
the label is known (1).

You then enter your test inputs, which features are preprocessed as the training ones and the 
last two columns appear to be set to 0 for unknown label.

These 102 dimensional vectors are linearly embedded into 512 dimensional vectors that will be
fed to the transformers. Then the usual transformer pipeline follows:

- LayerNorm
- Multi-head self-attention (4 heads)
- Residual
- LayerNorm
- Feed-forward (1024 hidden nodes)
- Residual

For the attention mask all training samples attend to each other and test samples only attend
to training samples. Feed-forward is done by all tokens.

After 12 transformers like this the final test tokens appear to be passed through a linear layer 
$(512\rightarrow C)$ for $C\leq 10$ and softmax is applied to produce the final label distribution 
for each test sample.

As you can see the structure is quite simple and there is nothing magical behind it, instead 
the ICL behaviour raises from the theoretical foundations we laid down and the training with
an appropriate **prior** generation choice.

## Train your own TabPFN

> *Our prior also takes ideas from Occam's razor: simpler SCMs and BNNs 
> (with fewer parameters) have a higher likelihood.*

> *Minimizing negative log likelihood approximates the true Bayesian 
> posterior predictive distribution*

When creating your own TabPFN this will most likely be the trickiest part. Luckly the 
theoretical framework is already laid out so you don't have to worry about existential 
questions, but designing a Dataset generator that encapsulates the entirety of the possible 
data relationships you want your pretrained model to infer is a tricky point, and you have
to be careful with the way you design it.

In the paper the choose a mixed approach to generating their **prior** between Structural 
Causal Models (SCMs) and Bayesian Neural Networks (BNNs). For a detailed explanation on why 
they chose this distribution and how exactly they sample datasets I recommend checking the 
lenghty Appendix C section of the paper.

Once they have their model generator they follow the training algorithm for a total of 18000 
steps with batches of 512 newly generated Datasets each. They use Adam optimizer with 
linear-warmup and cosine annealing. For each training they tested a set of 3 learning rates, 
{.001, .0003, .0001}, and used the one with the lowest final training loss.

## TabPFN limitations

Since most of the limitations mentioned in the 2023 paper are most likely addressed on the 
newer versions I will avoid talking about those right now.

But a limitation I potentially see arising and that is also hinted in the paper is one of 
specialization. TabPFN seems to want to solve everything with one algorithm, and for a more 
specific set of tabular datasets maybe a different choice of **prior** and even NN architecture 
might be more desirable for solving them or might require much less computation.

A good example (how convenient) could be market data. Most datasets on market analysis are 
dominated by noise, correlation between features weak and sparse and your objective is not 
to predict the exact price but to make a conservative portfolio strategy that produces returns 
with low volatility. 

Having this considerations in mind maybe a PFN specifically trained for this kind of tasks 
could outperform TabPFN with much less amount of compute. Although this is just an idea and 
needs a lot of thought and experimentation to back it up.

## Day 2 Checklist

- ✅ Understand PFNs, in-context learning framing for tabular data, support/query formulation.
- ✅ Convert to interview-ready notes.
- ✅ Deliverable: "notes/" day 2 doc.

## Frozen decisions

- TabPFN is actually quite cool
- TabPFN is actually quite complicated

## Open questions

- How do newer versions of TabPFN compare to their predecessor and what are the differences?
- How are the new versions of TabPFN used and calibrated on a professional setting?
- Can I replicate a small PFN pretrained NN for testing given my compute limitations?
- How will TabPFN perform when faced with noisy market data?