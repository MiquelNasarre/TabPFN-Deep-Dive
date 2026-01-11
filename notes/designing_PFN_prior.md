# Notes on `my_small_prior.py`

As we have discussed at length during the theory notes, the **prior** is an essential part of 
the design of a PFN, this is because it defines the function our NN is trying to approximate,
and therefore it fully conditions the kind of relations our PFN will look for in the data.

This is exactly why when building a **prior** you need to make it extensive enough to contain all 
the possible functions you would like your model to find, or at least a significant approximation
to them. This creates a relation between how diverse and complex your **prior** is and how expressive
your NN architecture needs to be to be able to encompass that diversity of causal relations between 
data.

## **Prior** design

Since our PFN architecture is quite small and this is just a proof-of-concept we do not need a very
complex design, so we will follow a similar structure for the **prior** to the one used for the 
original TabPFN training, which we covered on day 2.

### The MLP

So how do you define a SCM? First you start by sampling a random number of layers with a random 
number of nodes each, you create the matrices and biases for the linear projection but then drop 
the connections between nodes at random by a given amount. This allows for all kind of connectivities
between nodes, creating a random graph.

Then you assign a random non-linearity for each layer and also a random standard deviation for the noise 
that will be added to each node during the forward pass.

### The forward pass

Now we have an MLP — with randomized sparse connectivity, different non-linearities and noise — we can 
run forward passes on. So we create a random tensor of shape $(B, S, I)$ where $B$ is the micro-batch 
size used for training, $S$ is the sum of the training row and test rows we want to generate, and $I$ 
is the input layer dimension of the MLP.

We send this tensor through the MLP. For each layer the linear projection is applied, consisting of a 
matrix multiplication with the dropped connections and the addition of the biases. Then we apply the 
non-linearity and finally the random noise to each node.

Once the entire forward pass is finished all the layers are stacked into a big list of nodes, and it
outputs a tensor of shape $(B, S, N_{nodes})$,

### Choosing features and normalization

Once the MLP output is received a random set of nodes are chosen to be the features, and a single node
is chosen to be the target. This method ensures a lot of diversity in the causality of features and 
target, since a random causal graph is generated, and the features and target are just unordered nodes
of that graph. Although the same MLP is used for the entire micro-batch, distinct features and target 
are chosen for each dataset, making the **prior** more rich by containing different features and target
of the same causal graph. 

Now we have $B$ datasets stored in a tensor of size $(B, S, F+1)$ this is almost ready to be fed to 
the NN for training, but our NN expects normalized features and target, so we normalize each feature 
and target independently. This can be considered a leak to the test set, since it gives information 
about the test rows of the target, nevertheless it is not a big leak so I don't consider it much of 
a problem for a proof-of-concept.

The tensor is then split into the four tensors that will be used for training.

## Default variables

Choosing the exact settings for the **prior** is quite a tricky part, since it defines exactly how 
big, how noisy and how complex you want your datasets to be, and it will reflect the type of datasets 
your PFN will be expecting on the future.

Since our model will be tested on real market data then, it would be foolish to not give it noise, 
or to make it learn really complex representations of the input data, since mostly what we will be 
actually looking for is for faint causal relationships between the features and the target with lots 
of noise.

Another thing to consider is our current capabilities, unfortunately my GPU can only do so much, with 
2GB of VRAM it barely fits a training pass with micro-batch size 4, 24 features, 256 training rows and 
128 test rows. So those will be our limits on that regard. Also such small datasets cannot allow for 
much complexity otherwise the relations in the data are too faint to be significant, so that will 
affect how we choose the rest of the parameters.

You can check the actual default parameters used in the file itself, they are chosen with all the 
previous considerations in mind, but purely based on instinct. Nevertheless, since I am writing this
after some training has been done I can confirm the model actually learns with them.

## How do we make it better?

If more compute was available some improvements could clearly be done to the **prior** starting by 
the size, most datasets are much larger than 24 features and 256 rows, training with bigger datasets
would really teach the NN to work with what it has, and that if it has more data it can use that to 
its advantage to find more complex relations.

Other big improvements would come from making the data more similar to real world data, for starters
missing features are not accounted for, but also the entire list of variables has been chosen by me 
on pure intuition, that is clearly not the best metric from which to build our **prior**, instead 
multiple models could be trained on different configurations and then tested on real world data to 
decide variables on a more reasonable way.

There are many ways to make the data better, maybe adding different distributions for some nodes, 
like uniform distributions or sine waves, or adding different ways of creating SCMs and combining 
them, or even do some post-training on real world data like Real-TabPFN-2.5 did.

Creating priors tailored to your data should be a well thought task, and there are many ways to 
improve on our current generation, but it is always important to make sure your **prior** stays 
true to how data is in the real world, and behaves similar to the kind of data that will be 
presented to our PFN in the future.

## Conclusions

Currently the **prior** looks quite solid for a proof-of-concept and that is shown by the model 
learning and being able to perform some basic tests available at `my_trained_PFN_tests.ipynb`. It 
is also true that it might be a bit too complex for the model to learn some basic data relations,
specially given the small dataset sizes we are working with. 

Once the full training is done the model will be tested on the Hull Tactical data, and if it does 
not show too good of a performance I will consider simplifying the **prior** to see if the model
can learn better with simpler tasks.

