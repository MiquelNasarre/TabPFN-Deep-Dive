# Day 04.1 — How to train your TabPFN-2.5

The entirety of day 4 was supposed to be about discovering how TabPFN-2.5 was trained and 
dissecting its **prior** and training methods compared with the original ones from the 2023
paper.

Much to my despair the training code and details for TabPFN-2.5 are proprietary, and not much 
public information is known about it. Nevertheless we can make some educated guesses on how 
the priors were generated and the training methods used. So based on available information 
that is exactly what we are going to do here.

## The original TabPFN **prior**

> *Our prior also takes ideas from Occam's razor: simpler SCMs and BNNs 
> (with fewer parameters) have a higher likelihood.*

As mentioned on the day 2 report the prior in arguably the most important part of a PFN, 
this defines how it will later produce its models, since the function the PFN is trying 
to approximate is fully characterized by the **prior**. Therefore we will first revise 
how the original TabPFN **prior** was generated according to the 2023 paper.

Original ICLR 2023 paper (Appendix C for **prior** generation): <https://arxiv.org/abs/2207.01848>

### The sampling algorithm

Quoting directly:

We instantiate a subfamily of DAGs that can be efficiently sampled from by starting with a MLP 
architecture and dropping weights from it. That is, to sample a dataset with $k$ features and n 
samples from our prior we perform the following steps for each dataset:

(1) We sample the number of MLP layers $\ell ∼ p(\ell)$ and nodes $h ∼ p(h)$ and sample a graph $G(Z,E)$ structured like an $\ell$-layered MLP with hidden size $h$.

(2) We sample weights for each Edge $E_{ij}$ as $W_{i,j} ∼ pw(·)$.

(3) We drop a random set of edges $e \in E$ to yield a random DAG.

(4) We sample a set of $k$ feature nodes $N_x$ and a label node $N_y$ from the nodes $Z$.

(5) We sample the noise distributions $p(\epsilon) ∼ p(p(\epsilon))$ from a meta-distribution. This yields an SCM, with all $f_i$’s instantiated as random affine mappings followed by an activation. Each $z_i$ corresponds to a sparsely connected neuron in the MLP.

With the above parameters fixed, we perform the following steps for each member of the dataset:

(1) We sample noise variables $\epsilon_i$ from their specific distributions.

(2) We compute the value of all $z \in Z$ with $z_i = a((P j\in PAG(i) E_{ij} z_j ) + \epsilon_i)$.

(3) We retrieve the values at the feature nodes Nx and the output node Ny and return them.

We sample one activation function a per dataset from {$Tanh$, $LeakyReLU$, $ELU$, $Identity$} (Nair
and Hinton, 2010). The sampling scheme for the number of layers $p(l)$ and nodes $p(h)$ is designed to
follow a discretized noisy log-normal distribution, $p(\epsilon)$ is a noisy log-normal distribution and 
the dropout rate follows a beta distribution.

This is the Structured Causal Model (SCM) generator as described in the original TabPFN 
paper. This describes the creation of a simple MLP, where neuron connections are dropped at random,
random noise is added to each neuron, a random ativation function is chosen per dataset and the
nodes that will be used for the table features and targets are chosen at random from the resulting 
graph.

The paper also outlines the use of Bayesian Neural Networks, and other methods used to make their datasets
resemble even more real-world datasets, most remarkably their section on correlated features, where they 
explain how they use "Blockwise feature sampling" to better mimic real-world tabular data correlations, 
giving a compelling visual example of the veracity of their claims. Other techniques are applied to enrich
the **prior** generation, see the paper for more information.

This architecture alone can produce a wide variety of causal models, reason why the original TabPFN 
design was so successful despite their compute limitation and basic NN design. But it also has severe
limitations that can also be found in the paper itself.

### The original **prior** limitations

- **NaN handling**: Tabular data usually contains missing cells, fact not accounted in the original TabPFN architecture.

- **Irrelevant features**: Despite the prior containing a wide variety of functions it fails to capture
the amount of irrelevant features found in real-world tabular data, making the original TabPFN not that 
robust to irrelevant features.

- **Small scale**: It is apparent that with their compute limitations they could only achieve so much,
allowing only for 100 features and trained only to 1000 row data, not performing as well out of these limitations.

- **Classification only**: The target was always discretized, so no regression was performed by the original TabPFN.

- **Categorical Features**: Despite including them in their prior generation they failed to capture their essence, 
seeing a clear performance decrease when those were present on their datasets.

## The TabPFN-2.5 **prior**

Having what we just discussed in mind, lets review again the TabPFN-2.5 technical report and see 
what we can deduce from their **prior** generation.

TabPFN-2.5 Technical report: <https://arxiv.org/abs/2511.08667>

> **Data:** We improved our prior data generation substantially, broadened the set of distributions and
> scaled up to more data points and more features, while keeping the prediction tasks difficult. Like the
> original TabPFNv2, TabPFN-2.5 is trained purely on synthetically generated data. We also release a
> version that is fine-tuned on real data following Real-TabPFN. It is trained on a curated corpus of
> 43 real-world tabular datasets sourced from OpenML and Kaggle, ...

It is difficult to infer what those **prior** improvements are exactly, but knowing the previous limitations 
it is safe to assume they added measures to improve all of them, since throughout the report they proudly claim 
robustness on all the prevoiusly mentioned limitations. 

The model appears to have trained on *more than a hundred million synthetic datasets*, given the scaling difference
it is safe to assume more advanced techniques are used to further group features and express causality between 
them in a way that further resembles real world data.

Their Real-TabPFN-2.5 model is also a very important note, it clearly outperforms the original, and that is just
by fine tuning with $43$ real datasets, this tells us that the complexity and nuances of real data are difficult
to reproduce synthetically, and real world data is still a valuable asset for training.

Despite all of those changes I do believe their **prior** generation follows a similar SCM foundation to the 
original one, since the core idea behind is quite solid, other architectures have definitely been tried and 
added on top of it, but I do believe the original foundation persists.

## Hyperparameter Tuning

For this particular topic not much digging was needed, since they proudly expose their clever method in the 
technical report, therefore I'll just quote:

TabPFN’s hyperparameter space spans architectural, training, and prior-data parameters, making exhaustive 
grid search computationally infeasible. To explore this space efficiently, we adopted a surrogate-based 
optimization strategy. We first trained $\approx 100$ models on a broad but sparse grid of hyperparameter 
configurations drawn from plausible prior ranges and evaluated them on a curated in-house validation suite, 
producing a compact set of hyperparameter–performance pairs.

With $\sim 50$ hyperparameters and only $100$ datapoints, direct interpolation was prone to overfitting.
We therefore used a regression model well-suited for data-scarce structured prediction—our previous
TabPFNv2 model—as a surrogate to predict validation performance over a denser grid of $10,000$ configurations. 
This self-referential “TabPFN-tunes-TabPFN” strategy efficiently surfaced promising regions of the search space 
for full, compute-intensive training runs.

## Training a Bayesian regressor

One of the coolest things of TabPFN-2.5 its their regressor, and the fact that it does not output a number 
but instead — on pure Bayesian fashion — it outputs a probability distribution for the target. But after 
analizing the foundations of the model this is not only a viable approach, but the most logical one.

> *Minimizing negative log likelihood approximates the true Bayesian 
> posterior predictive distribution*

This quote already appeared on day 2 but I find it too strong to ignore, what this proven fact — within 
the PFN objectinve of approximating a PPD generator based on a prior — is telling us is that the best way 
of deriving the Bayesian Posterior Predictive Distribution we described on day 2 as $\pi(·|x,D)$ is by 
minimizing negative log likelihood.

This gives us an immediate road map for classifiers, since NLL is widely used and we can simply optimize 
our NN using SGD with this loss function, but what about regressors?

The TabPFN answer is simple and quite pretty, instead of outputting a single guess for regression, let's 
actually recreate the distribution we are trying to mimic, that is evident since they function $\pi$ we are
trying to mimic outputs PPDs not numbers. How do we do that? Let's discretize the real number line in 
$5000$ buckets and output logits for each bucket (as we saw yesterday). This way we can use NLL as our 
loss function for those logits, and keep true to the Bayesian foundations of our model.

Although the exact loss and regularizations are not publicly available, giving the theoretical foundations
and their complete lack of mention of other methods or applied regularizations, it is safe to assume this 
is exactly how they trained their regressor, simple NLL minimization over the logits.

NLL is known for producing spiked guesses in many supervised learning settings, and one might think that
regularization would be needed to avoid that for TabPFN, but it is most likely the case that the lack of 
spiky guesses from their model does not come from strong regularization of NLL, but instead is due to its 
theoretical foundations and its rich ecosystem of synthetic datasets, carefully engineered to produce
real-world noisy results, that will best be described by a wider probability distribution.

## Conclusions

Not having public data on the training of TabPFN-2.5 is clearly a pushback when trying to understand 
how it works and how to build similar models in the future. Nevertheless, it is not a complete blocker.

Given the publicly available claims and what we know of the old models we can make some educated guesses
on how the training was done. With those guesses we have enough information to reproduce a functional 
PFN model, maybe not as sophisticated as TabPFN, but with rich capabilities and with a **prior** that 
will most likely resemble the one actually used for TabPFN-2.5.

What follows for today is the missing piece of TabPFN-2.5, this being its preprocessing and 
post-processing of the variables, important part to understand its true capabilities.