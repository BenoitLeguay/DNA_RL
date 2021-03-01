# Algorithm description

## 1. Markovian Decision process

For this algorithm, we are going to optimize a DNA sequence iteratively. 

A Markov Decision Process (MDP) is a 6-tuple $(S,A,P,R,γ,D)$, where $S$ is the state space of the process,$A$ is  the  action  space, $P$ is a Markovian  transition model $P(s′|s,a)$ denotes the probability of a transition to state $s′$ when taking action $a$ in state $s$, $R$ is a reward function $R(s,a)$ is the expected reward for taking action $a$ in state $s$, $γ∈(0,1)$ is a discount factor for future rewards, and $D$ is the initial state distribution. A deterministic policy $π$ for an MDP is a mapping $π:S → A$ from states to actions; $π(s)$ denotes the action choice in state $s$.



#### State

Our state $s_t$ is going to be the current sequence to be optimized at time $t$, let's name it $S_{opt}$. The latter has a fixed length named $L_{opt}$. To perform modifications, the agent will use crossover and thus we'll need a **unmutable** DNA sequence called $S_{co}$ with $L_{co} \geqslant L_{opt}$. This sequence could be define randomly or maybe in a more judicious manner, such as the concatenation of all combination of ${A, T, G, C}$ in a range of k (*e.g k  from 1 to 5*). In this vision, $S_{co}$ would not be in the state $s_t$ as the modification won't affect the sequence.

Alternatively, the state $s_{t}$ could be seen as the concatenation of both $S_{opt}$ and $S_{co}$ with $L_{opt} = L_{co}$. That would allow multiple enhancements for larger computation cost. First crossovers could modify $S_{co}$ permitting a wider modification range. Moreover, that makes us free of choosing arbitrarily a $S_{co}$. 



#### Action and Transition function

The action is defined in a **discrete 3-dimensional** space such as $a_t = (l, i_{opt}, i_{co})$, where $l$ is the crossover length, $i_{opt}$ and $i_{co}$ are respectively the crossover starting point (*i.e the index*) for $S_{opt}$ and $S_{co}$. In that way we can perform all types of crossover. The transition function $P(s'|s, a)$ is then purely deterministic. I describe the function approximation architecture in section 3. 



#### Reward

The reward function is handle by a ***Oracle***. This neural network takes a DNA sequence as input and output its estimation of how good is it. So $r_t = Oracle(s_{t})$ with $r_t \in [0, 1]$. 



### 2. Algorithm

1. initialize policy parameter $\theta_0$, value function parameter $w_0$, and both step size $\alpha_a, \alpha_c$ . Let  $\gamma$ be our discount factor, $m$ be our population size and $r_{target}$  be our score target 

2. Initialize $s_0$ (*i.e a random series of shape $(L_{opt}, )$*)

3. for $t$$  = 0, 1, 2, 3 ... $T 

   1. the actor selects action $a_t$, sampling from $\hat{\pi}(. | s_t)$.

   2. perform $a_t$, observe $s_{t+1}$.

   3. compute $r_{t+1}$ with the *Oracle* such that $r_t = Oracle(s_{t})$.

   4. if $r_t > r_{target}$ : *break from loop*

   5. Using critic, compute both $\hat{V}(s_t)$ and $\hat{V}(s_{t+1})$.

   6. Compute the TD target: $\delta_t = r_{t+1} + \gamma * \hat{V}(s_{t+1})$ and the TD error: $\delta_e=\delta_t - \hat{V}(s_t)$.

   7. update actor minimizing $L_a = - log(\hat{\pi}(a_t|s_t)) * \delta_e$. (*Entropy penalty is discussed in 5.*)

      1. $\theta_{t+1} = \theta_t +\alpha_a * \frac{\partial L_a}{\partial \theta_t} $.

      update critic minimizing $L_c=(\delta_t - \hat{V}(s_t))^2$.

      1. $w_{t+1} = w_t +\alpha_c * \frac{\partial L_c}{\partial w_t}$.

   8. $s_t = s_{t+1}$.
   
      

## 3. Function Approximation

In this section we will talk about both **critic** and **actor** networks. 

#### Actor

###### Autoregressive

In this section, we will assume $|a_1|=· · ·=|a_d|=L_{opt}$  with $d=3$. In the section 5, we talk about dealing with different sub-action's dimension size.

The actor needs to generate a probability distribution over the set of actions $\mathcal{A}$. In a 1-dimensional action space setting, the classical approach is to use a SoftMax output of dimension $|\mathcal{A}|$. In our specific case, $\mathcal{A} \in \mathbb{N}^3$ with $a_i < L_{opt} \> \forall \> a_i \>in \> \mathcal{A} $. That implies, generating a multivariate probability distribution over the action space. The stake of the model to use is real here for computational efficiency purpose.  

Indeed, when dealing with multidimensional discrete action space, the number of possible action is exponential with respect to the number of dimension. Let's recall that our action is defined by the 3-tuple $a_t = (l, i_{opt}, i_{co})$, let's imagine we are going perform crossover on $S_{opt}$ with $L_{opt}=50$. Each of $l, i_{opt}, i_{co}$ can take a positive discrete value inferior to $50$.  Since using all possible combination with order and repetition (and thus $50^3$ different actions) would force the neural network to handle a 125000 output layer shape, I'm quite unwilling to go this way. 

[Recent work](https://arxiv.org/pdf/1806.00589.pdf) have shown autoregressive model can be used to sample sequentially $A_i = (a_1, a_2, ..., a_i)$ from our policy. In that way sampling from model only requires summing over $O(3*L_{opt})$ effort whereas the aforementioned model requires $O((L_{opt})^3)$. 

![1591061401846](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/1591061401846.png)



![1591061487464](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/1591061487464.png)





As they are quite easy to implement with nowadays deep learning framework, the goal here would be to test both RNN and MMDP as autoregressive model for the **actor**.



###### N independent Actors (Baseline)

Since we want to compare our autoregressive model with a baseline, I choose to implement a trivial actor composed of N (=3) actors each of them powered by its personal neural network. Thus each of them estimates a probability distribution over 1 action dimension (crossover length, and both starting points).  They really are independent in the sense that no one is affected by one another in the learning process, they all use the same reward to update their weights independently. Of course, the overall result should be poor since there is no learning between the different action dimensions. Indeed choosing the crossover length regardless the starting point is intuitively a bad way of doing. Though, this approach will serve as a baseline. 



![](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/n_independent_actors.png)



###### N actions Actor

Another way of predicting multivariate distribution is to use a single neural network with multiple independent softmax on the last layer. Each piece of action is now generated with regards to the 2 other ones. This architecture could be interesting to compare with the autoregressive one. 



![n_actions_actor](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/n_actions_actor.png)



#### Critic

It is a much more easier network to handle, since it only estimates the value of a particular state $s_t$. A simple neural network taking the state $s_t$ as input and outputting a scalar will make the job.



## 4. Feature engineering

Dealing with large state space can slow learning, and more precisely generalization over state. One way to manage this difficulty is to learn good latent representations to construct the states with meaningful information. The *Oracle* has been trained to grasp the DNA architecture, by taking a latent space of this network we could find a better representation of our state. 

![1591104437286](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/1591104437286.png)

## 5. Exploration and Entropy

Good exploration method is a important point with large action space, the entropy penalty is quite efficient and easily scalable to our actor network. The entropy is a measure of uncertainty within a certain probabilistic distribution, it is the average “element of surprise” or amount of information when drawing from the probability distribution. When  the agent is learning its policy and an action returns a positive  reward for a state, it might happen that the agent will always use this  action in the future because it knows it produced *some*  positive reward. There could exist another action that yields a much  higher reward, but the agent will never try it because it will just  exploit what it has already learned. This means the agent can get stuck  in a local optimum because of not exploring the behavior of other  actions and never finding the global optimum. This is where entropy comes handy: we can use entropy to encourage exploration and avoid getting stuck in local optima.



**Entropy**

To abbreviate notations, we write $p_{\theta}(a)$ for $\pi_{\theta}(a|s_t)$ and $a_i$ for $(a_1, a_2, .., a_i)$. We consider auto-regressive models whereby the sample components $a_i, \> i = 1, 2, .., d$ are sequentially generated, with $d=2$ in our case. In particular, after obtaining $a_1, a_2, ..., a_{i-1} $, we will generate $a_i \in \mathcal{A}_i$ from some parameterized distribution $p_{\theta}(.|a_{i-1})$ defined over the one-dimensional set $\mathcal{A}_i$. After generating the distribution $p_{\theta}(.|a_{i-1}) \>\forall\> i$ and sample the action component $a_1, a_2, .., a_d$ sequentially, we then define $p_{\theta}(a) = \prod_{i=1}^{d}p_{\theta}(a_i|a_{i-1})$.

​	$H_{\theta_t} = - \sum_{a_i \in \mathcal{A}} p_{\theta}(a)log(p_{\theta}(a))$

​	$H_{\theta_t} = -\mathbb{E}_{A \sim p_{\theta}} [log\>p_{\theta}(A)]$

​	$H_{\theta_t} =-\sum_{i=1, .., d} \mathbb{E}_{A \sim p_{\theta}} [log\>p_{\theta}(A_i | A_{i-1})]$



**Crude unbiased estimator**

During training within an episode, for each state $st$, the policy generates an action $a$. We refer to this generated action as the episodic sample. A crude approximation of the entropy bonus is:

$H_{\theta}^{crude}=-log\>p_{\theta}(a)=-\sum_{i=1}^{d}p_{\theta}(a_i|a_{i-1})$

This approximation is an unbiased estimate of $H_{\theta}$ but its variance is likely to be large. To reduce the variance, we can generate multiple action samples when in $s_t$ and average the log action probabilities over the samples. However, generating a large number of samples is costly, especially when each sample is generated from a neural network, since each sample requires one additional forward pass.



**Smoothed Estimator**

This section proposes an alternative unbiased estimator for Hθ which only requires the one episodic sample and accounts for the entropy along each dimension of the action space:

​	$\tilde{H}_{\theta}(a) := -\sum_{i=1}^{d}\>\sum_{\alpha \in \mathcal{A}_i}\>p_{\theta}(\alpha|a_{i-1})\>log\>p_{\theta}(\alpha|a_{i-1})$

​	$\tilde{H}_{\theta}(a) = \sum_{i=1}^{d}\>H_{\theta}^{(i)}(a_{i-1})$

with

​	 $H_{\theta}^{(i)}(a_{i-1}) := -\sum_{\alpha \in \mathcal{A}_i}p_{\theta}(\alpha|a_{i-1})\>log\>p_{\theta}(\alpha|a_{i-1})$

which is the entropy of $\mathcal{A}_i$ conditioned on $a_{i−1}$. This estimate of the entropy bonus is computationally efficient since for each dimension $i$, we would need to obtain $p_θ(·|a_{i−1})$, its log and gradient anyway during training. We refer to this approximation as the smoothed entropy.



**Gradient Estimator**

To make a lighter document, I will not defined mathematically the gradient of the Entropy. The aforesaid mathematical aspect gives good insights on the entropy choice whereas the gradient formula here is less relevant. You can find the whole definition in [this paper](https://arxiv.org/pdf/1806.00589.pdf).



## 6. Differences in each action component dimension

We might want to restrict the crossover size and thus $L_{opt}$ to a certain range (*e.g from 15 to 20*). This would improve learning drastically without restraining too much the learning spectrum. Indeed, a 1-sized cross over is really not the vision here nor a 40-sized which would have a substantial effect to the DNA sequences. 

The issue is that, as we are going to use a autoregressive model (*e.g LSTM network*), the output layer size will be set. The idea behind using inconsequential actions has been largly used in the RL community recently (*e.g AlphaGo*). So we are going to proceed using this 2 rules: 

- Not selectable actions are tremendously penalized

- Not selectable actions have no effect on the state

  

## 7. Asynchronous

For a better understanding, the algorithm describes in 2. does not mention the asynchronous aspect. A3C consists of **multiple independent agents** (networks) with their own weights, who interact with a different copy of the environment in parallel. Thus, they can explore a bigger part of the state-action space in much less time. The agents (or workers) are trained in parallel and update periodically a global network, which holds shared parameters. The updates are not happening simultaneously and that’s where the asynchronous comes from. After each update, the agents resets their parameters to those of the global network and continue their independent exploration and training for n steps until they update themselves again.

We see that the information flows not only from the agents to the global network but also between agents as each agent resets his weights by the global network, which has the information of all the other agents. 

![image-20200607190705345](/home/benoit/Documents/work/RL_DNA/paper_proposition/photos/image-20200607190705345.png)



*RESSOURCE*

- https://arxiv.org/pdf/1806.00589.pdf
- https://theaisummer.com/Actor_critics/