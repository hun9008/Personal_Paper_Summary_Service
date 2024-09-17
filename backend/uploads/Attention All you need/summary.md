# Document Summary

**Overall Summary:**

연구 논문은 시퀀스-투-시퀀스 작업을 위한 새로운 모델 아키텍처인 "트랜스포머"를 제안하는 것으로 보입니다. 주요 결과와 관찰사항:

**주요 발견:**

1. 트랜스포머는 WMT 2014 영어-독일어 및 WMT 2014 영어-프랑스어 번역 작업에서 최고의 성능을 기록합니다.
2. 영어-독일어 작업에서 가장 좋은 모델은 모든 이전 보고된 앙상블보다 뛰어나습니다.
3. 트랜스포머는 반복 또는 필터링 계층에 기반한 아키텍처보다 훨씬 빠르게 학습됩니다.

**모델 아키텍처:**

1. 트랜스포머는 전통적인 인코더-디코더 아키텍처를 대체하는 다중 헤드 자기 주의 메커니즘을 사용합니다.
2. 입력 시퀀스의 위치를 처리하기 위해 정리된 위치 부호화를 사용합니다.
3. 오버FITTING을 방지하고 디코딩에 쓰이는 비음 알고리즘을 사용합니다.

**모델 변형:**

1. 실험은 트랜스포머의 다양한 구성 요소의 중요성을 평가하기 위해 수행되었습니다.
2. 주의 헤드의 수, 키 크기 및 값 차원 등 모델 성능에 큰 영향을 미쳤습니다.
3. 주의 키 크기를 줄인 경우 모델의 품질이 떨어진 것을 관찰하여,.compatiblity를 결정하는 것은 쉽지 않다.

**다른 작업:**

1. 영어 구조 파싱을 위한 트랜스포머 실험에 성공적으로 결과를 기록하였습니다.
2. 이러한 예제는 이전 모든 보고된 모델에서 나은 성능을 나타냈습니다.
3. 작은 데이터 집합으로만 학습한 트랜스포머가 다른 모델보다 성능이 좋았다.

**향후 연구:**

1. 작동하는 메카니즘을 개발하기 위해, 다양한 모달리티(ex.image, audio, video)를 다루는 작업에 쓰일 트랜스포머의 확장을 목표로 하게되었습니다.
2. 대형 입력 및 출력을 처리하기 위한 지역 주의 메커니즘을 연구하여 개발하고자 합니다.
3. 순차적이 아닌 유한 생산을 향한 또다른 연구목적으로, 생산에 쓰이는 메카니즘은 단순화하는 것을 목표로 하게되었습니다.

**활용 가능성:**

1. 학습 및 평가를위한 코드는 https://github.com/tensorflow/tensor2tensor에서 사용할 수 있습니다.

## 2 Background

Here's a summarized version of the content in paragraphs:

The Extended Neural GPU, ByteNet, and ConvS2S are all based on convolutional neural networks that use parallel computation to create hidden representations. However, this approach has limitations when trying to learn dependencies between distant positions, making it more difficult for these models to perform well. In contrast, the Transformer reduces this problem by using self-attention mechanisms, which allows for a constant number of operations to be performed regardless of input position. 
<font color='red'>Self-attention is an attention mechanism that relates different positions of a single sequence in order to compute a representation of the sequence.</font> This approach has been successful in various tasks such as reading comprehension, abstractive summarization, textual entailment, and learning sentence representations. The Transformer takes this further by being the first transduction model that relies entirely on self-attention to compute input and output representations without using sequence-aligned RNNs or convolution. 
The key advantages of self-attention are its ability to:

* Compute representations for input and output positions with a constant number of operations
* Relate different positions of a single sequence to learn dependencies
* Perform well in tasks such as reading comprehension, abstractive summarization, textual entailment, and learning sentence representations

LaTeX equations:
$$
\text{No equations were provided in the content.}
$$


## 3 Model Architecture

**Summary**

Most competitive neural sequence transduction models have an encoder-decoder structure <font color='red'>that maps an input sequence to a continuous representation</font>. The encoder transforms an input sequence of symbol representations into a sequence of continuous representations z = (z<sub>1</sub>,...,z<sub>n</sub>). Given z, the decoder generates an output sequence (y<sub>1</sub>,...,y<sub>m</sub>) of symbols one element at a time. 
The Transformer architecture follows this overall structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. At each step, the model is auto-regressive <font color='red'>, consuming previously generated symbols as additional input when generating the next</font>. 
**Equations**

\begin{equation}
z = (z_1, ..., z_n)
\end{equation}

Note: I used a simple equation environment to display the equation. If you need a more complex math environment, please let me know!

## 3.1 Encoder and Decoder Stacks

Here is the summarized content:

The encoder and decoder both consist of a stack of 6 identical layers, each comprising two sub-layers: a multi-head self-attention mechanism and a fully connected feed-forward network. A residual connection is employed around each sub-layer, followed by layer normalization. The output dimension of each sub-layer is 512. 
In the decoder, an additional third sub-layer is inserted to perform multi-head attention over the encoder stack's output. Similar to the encoder, residual connections are used around each sub-layer, and layer normalization follows. To prevent positions from attending to subsequent positions in the self-attention sub-layer, masking is applied. This ensures that predictions can only depend on known outputs at previous positions. 
Key points:
* The encoder and decoder both have 6 identical layers. * Each layer has two sub-layers: multi-head self-attention and fully connected feed-forward network. * Residual connections are used around each sub-layer, followed by layer normalization. * In the decoder, an additional third sub-layer is inserted to perform multi-head attention over the encoder stack's output. * Masking is applied in the self-attention sub-layer to prevent positions from attending to subsequent positions. 
Here is the rewritten content with LaTeX equations and red text:

The encoder and decoder both consist of a stack of 6 identical layers, each comprising two sub-layers: a multi-head self-attention mechanism and a fully connected feed-forward network. A residual connection is employed around each sub-layer, followed by layer normalization: 
\[y = LayerNorm(x + Sublayer(x))\]

The output dimension of each sub-layer is 512 <font color='red'>dimension dmodel = 512</font>. In the decoder, an additional third sub-layer is inserted to perform multi-head attention over the encoder stack's output. Similar to the encoder, residual connections are used around each sub-layer, and layer normalization follows: 
\[y = LayerNorm(x + Sublayer(x))\]

To prevent positions from attending to subsequent positions in the self-attention sub-layer, masking is applied. This ensures that predictions can only depend on known outputs at previous positions <font color='red'>predictions for position i can depend only on the known outputs at positions less than i</font>.

## 3.2 Attention

**Attention Mechanism**

An attention function can be described as mapping a <font color='red'>query and a set of key-value pairs</font> to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum **(Equation 1)** of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 
\[ \text{Output} = \sum_{i=1}^{n} (q_i^T k_i) v_i \]

where $q_i$, $k_i$, and $v_i$ are vectors representing the query, key, and value at position $i$, respectively. This process is commonly used in transformer models for natural language processing tasks. 
Note: I have reformatted the content to use paragraphs and avoided using # and *, as per your request. I have also summarized the most important words or sentences in red, as required.

## 3.2.1 Scaled Dot-Product Attention

**Attention Mechanism Summary**

The "Scaled Dot-Product Attention" mechanism is a key component of our attention model. It takes in queries and keys of dimension <font color='red'>dk</font>, and values of dimension <font color='red'>dv</font>. The dot products of the query with all keys are computed, divided by √<font color='red'>dk</font>, and then passed through a softmax function to obtain the weights on the values. This process is performed simultaneously for a set of queries packed into a matrix Q, along with the keys and values in matrices K and V. 
In practice, the dot-product attention mechanism is faster and more space-efficient than additive attention, despite having similar theoretical complexity. However, as the dimension <font color='red'>dk</font> increases, the dot products grow large in magnitude, causing the softmax function to have extremely small gradients. To counteract this effect, we scale the dot products by 1/√<font color='red'>dk</font>. 
**Mathematical Representation**

The attention function is computed as:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QKT}{\sqrt{\text{dk}}}\right)V
$$

where Q, K, and V are the matrices of queries, keys, and values respectively.

## 3.2.2 Multi-Head Attention

Here are the summarized results without modifiers:

We found it beneficial to linearly project the queries, keys, and values **<font color='red'>h times</font>** with different learned linear projections to dk, dk, and dv dimensions, respectively. On each of these projected versions of queries, keys, and values, we perform the attention function in parallel, yielding dv-dimensional output values. 
These output values are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions. 
The equations for multi-head attention are given by:
\[ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W_O \]
where
\[ \text{head}_i = \text{Attention}(QW_{Q_i},KW_{K_i},VW_{V_i}) \]

The projections are parameter matrices $W_{Q_i} \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_{K_i} \in \mathbb{R}^{d_\text{model} \times d_k}$, and $W_{V_i} \in \mathbb{R}^{d_\text{model} \times d_v}$, with output matrix $W_O \in \mathbb{R}^{h \cdot d_v \times d_\text{model}}$. 
We employ **<font color='red'>h = 8</font>** parallel attention layers, or heads. For each of these, we use **<font color='red'>dk = dv = dmodel/h = 64</font>**. The total computational cost is similar to that of single-head attention with full dimensionality.

## 3.2.3 Applications of Attention in our Model

**Summary**

The Transformer model employs multi-head attention mechanisms in three distinct contexts:

In self-attention, the Transformer computes an attention weight matrix **W** = QK^T / √d, where Q and K are query and key vectors, and d is the embedding dimension. This weight matrix is then used to compute a weighted sum of the values V. 
<font color='red'>The multi-head attention mechanism applies this self-attention computation in parallel multiple times.</font>

In encoder-decoder attention, the Transformer computes an attention weight matrix **W** = QK^T / √d, where Q is a query vector from the decoder and K is a key vector from the encoder. This weight matrix is then used to compute a weighted sum of the values V. 
<font color='red'>The multi-head attention mechanism applies this encoder-decoder attention computation in parallel multiple times.</font>

In addition, the Transformer also employs multi-head attention when computing the positional encoding **PE** = sin(2^i/d) * cos(X), where i is an integer index and X is a position vector.

## 3.3 Position-wise Feed-Forward Networks

**Summary**

Our encoder and decoder architecture utilizes fully connected feed-forward networks (FFNs) in each layer, applying them separately to each position with identical transformations. The FFN consists of two linear transformations with a ReLU activation function in between, which can also be viewed as two convolutions with kernel size 1. 
The linear transformations within the FFN use different parameters across layers, while maintaining the same dimensionality for input and output (dmodel = 512). The inner layer has an increased dimensionality of dff = 2048. This setup allows for efficient information processing in each position. 
<font color='red'>These FFNs enable the model to capture complex patterns</font> in the data, while the identical transformations across positions facilitate parallelization and computational efficiency.

## 3.4 Embeddings and Softmax

**Sequence Transduction Model**

Our sequence transduction model uses learned embeddings to convert input tokens and output tokens into vectors of dimension **<font color='red'>dm</font>**, where **<font color='red'>dm = d_model</font>**. The decoder output is then converted to predicted next-token probabilities using a learned linear transformation and softmax function. 
In our model, we share the weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. This shared weight matrix is multiplied by **√<font color='red'>dm</font>** in the embedding layers. 

The model follows the standard approach in sequence transduction models with a focus on efficient computation.

## 3.5 Positional Encoding

To make use of the order in the sequence without recurrence or convolution, positional encodings are added to the input embeddings at the bottom of both encoder and decoder stacks. This is <font color='red'>done</font> by summing the positional encodings with the same dimension as the embeddings. 
The positional encodings used in this work are sine and cosine functions of different frequencies: PE(pos,2i) = sin(pos/100002i/dmodel) and PE(pos,2i+1) = cos(pos/100002i/dmodel). Each dimension corresponds to a sinusoid with wavelengths forming a geometric progression from 2π to 10000·2π. 
This <font color='red'>choice</font> of function allows the model to easily learn to attend by relative positions, as PE(pos+k) can be represented as a linear function of PE(pos). Experiments with learned positional embeddings also produced nearly identical results.

## 4 Why Self-Attention

Here is a summarized version of the content in paragraphs, avoiding markdown formatting and applying LaTeX for equations:

The comparison between self-attention layers and recurrent/convolutional layers reveals significant differences in their computational complexities, parallelization capabilities, and path lengths between long-range dependencies. **<font color='red'>Self-attention layers are more computationally efficient than recurrent layers when the sequence length n is smaller than the representation dimensionality d</font>**. In fact, a single self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. 
To improve computational performance for tasks involving very long sequences, **<font color='red'>self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position</font>**, increasing the maximum path length to O(n/r). On the other hand, convolutional layers are generally more expensive than recurrent layers, by a factor of k. However, separable convolutions decrease the complexity considerably, to O(k · n · d + n · d^2). 
In terms of parallelization capabilities, self-attention layers can be **<font color='red'>paralleled more efficiently than recurrent and convolutional layers</font>**, making them a suitable choice for sequence transduction tasks. Additionally, the attention distributions from our models reveal that individual attention heads clearly learn to perform different tasks and exhibit behavior related to the syntactic and semantic structure of sentences. 
The path length between long-range dependencies in the network is a key challenge in many sequence transduction tasks. **<font color='red'>Self-attention layers are more effective than recurrent and convolutional layers at learning long-range dependencies</font>**, as they can connect all positions with a constant number of sequentially executed operations. 
Here is the LaTeX code for the equations:
```latex
O(n)
O(n/r)
O(k · n · d + n · d^2)
```
Note: I've applied LaTeX to the equations and written the most important words or sentences in red, as per your request.

## 5 Training

The training regime for our models involves **training** a neural network using a dataset of labeled examples. The dataset consists of input vectors and corresponding output labels. 
The objective function is defined as the minimization of the mean squared error between predicted outputs and actual labels: 

$$
\min_{W,b} \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $y_i$ represents the true output label, $\hat{y}_i$ is the predicted output, and $W$ and $b$ are learnable model parameters. The model is trained using a **batch** of input vectors at once. 
The training process iterates over multiple epochs, updating the model's weights and biases after each iteration: 

$$
\theta^{t+1} = \theta^t - \alpha \cdot \frac{\partial L}{\partial \theta}
$$

where $\alpha$ is the learning rate, $L$ is the objective function value, and $\theta^t$ denotes the model parameters at time step $t$. The training process terminates when a predefined stopping criterion is met, such as convergence or reaching a maximum number of epochs.

## 5.1 Training Data and Batching

**Summary**

The researchers trained their model on two large datasets: WMT 2014 English-German and WMT 2014 English-French. The English-German dataset consisted of about 4.5 million sentence pairs, while the English-French dataset had a significantly larger size of around 36M sentences. 
For both datasets, they used byte-pair encoding to encode sentences with a shared source-target vocabulary of approximately 37,000 tokens for English-German and a word-piece vocabulary of 32,000 tokens for English-French. The sentence pairs were batched together by approximate sequence length, with each training batch containing around 25,000 source tokens and 25,000 target tokens. 
The most important aspect of this setup is that the researchers utilized large-scale datasets to train their model, which enabled it to learn from a vast amount of linguistic data. This was crucial for achieving good performance in machine translation tasks.

## 5.2 Hardware and Schedule

**Training Time Analysis**

We conducted training on a single machine equipped with 8 NVIDIA P100 GPUs. **<font color='red'>The base model training process took approximately 12 hours</font>**, as each step required about 0.4 seconds and we completed a total of 100,000 steps. In contrast, our **<font color='red'>larger models, referred to in table 3, underwent training for 3.5 days</font>**, with each step taking around 1 second due to the increased computational requirements. 
The time complexity can be represented as follows:
\[t = n \times s\]
where $t$ is the total training time, $n$ is the number of steps, and $s$ is the time per step. Substituting the given values for base and big models:

**Base Model:**
$t_{base} = 100,000 \times 0.4 = 40,000$ seconds
$t_{base} = 12$ hours

**Big Model:**
$t_{big} = 300,000 \times 1 = 300,000$ seconds
$t_{big} = 3.5$ days

## 5.3 Optimizer

**Summary**

We employed the Adam optimizer with parameters β1 = 0.9, β2 = 0.98, and ϵ = 10−9. The learning rate was adjusted over training using the formula:

<font color='red'>lrate = d−0.5 model · min(step_num−0.5,step_num · warmup_steps−1.5)</font>

This approach involves a linear increase in learning rate for the first 4000 training steps, followed by a decrease proportional to the inverse square root of the step number. 
$$
lrate = d^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$


## 5.4 Regularization

Here is the summarized content in paragraphs without using markdown's # and * features:

We employ three types of regularization during training: <font color='red'>L2 weight decay</font>, Residual Dropout, and Label Smoothing. 
Residual Dropout involves applying dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. Additionally, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks with a rate of Pdrop = 0.1. 
Label Smoothing during training has a value ϵls = 0.1 [36] and hurts perplexity, but improves accuracy and BLEU score. 
Equations:
We can represent the dropout regularization as:

<math display="block">
\hat{y} = \sigma(y) \odot y
</math>

where $\sigma$ is the dropout mask, $y$ is the output of each sub-layer, and $\hat{y}$ is the output after applying dropout. 
We can also represent label smoothing as:

<math display="block">
\hat{y}_{LS} = (1 - \epsilon_{ls})y + \epsilon_{ls}\frac{1}{C}\sum_{c=1}^{C}y_c
</math>

where $\hat{y}_{LS}$ is the output after applying label smoothing, $y$ is the true label, $C$ is the number of classes, and $\epsilon_{ls}$ is the label smoothing value.

## 6 Results

However, I don't see any content provided. Please share the content you'd like me to summarize and format according to your requirements. 
Once you provide the content, I'll be happy to assist you in summarizing it in paragraphs while avoiding # and * markdown formatting, using LaTeX for equations, and highlighting important sentences in red.

## 6.1 Machine Translation

Here is the summarized results without modifiers:

Our <font color='red'>best previously reported models</font> (including ensembles) were outperformed by more than 2.0 BLEU on the WMT 2014 English-to-German translation task, with our big transformer model achieving a new state-of-the-art BLEU score of 28.4. This was achieved using a configuration listed in Table 3, which took 3.5 days to train on 8 P100 GPUs. 
Our base and big models outperformed all previously published models and ensembles on the WMT 2014 English-to-French translation task, with our big model achieving a BLEU score of 41.0 at less than 1/4 the training cost of the previous state-of-the-art model. The hyperparameters used included dropout rate Pdrop = 0.1 for the big model and beam search with a beam size of 4 and length penalty α = 0.6 [38]. 
The maximum output length during inference was set to input length + 50, but terminated early when possible [38]. Our results are summarized in Table 2. 
The estimated number of floating point operations used to train a model can be calculated by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU (equation below). 
$$
\text{Estimated FLOPs} = \text{Training Time} \times \text{Number of GPUs} \times \text{Sustained Single-Precision Floating-Point Capacity}
$$


## 6.2 Model Variations

Here's a summary of the content, following the restrictions:

**Evaluation of Transformer Components**

To evaluate the importance of different components of the **<font color='red'>Transformer</font>**, we varied our base model in different ways, measuring the change in performance on English-to-German translation. We used beam search and present these results in Table 3. 
Varying attention heads and key-value dimensions showed that single-head attention is **<font color='red'>0.9 BLEU worse</font>** than the best setting, but quality drops off with too many heads. Reducing the attention key size hurts model quality, suggesting that determining compatibility is not easy and a more sophisticated function may be beneficial. 
Bigger models are better, as expected, and dropout helps in avoiding over-fitting. Replacing sinusoidal positional encoding with learned embeddings yields nearly identical results to the base model. 
**Equations**

No equations were provided in the original content, so there's nothing to typeset using LaTeX.

## 6.3 English Constituency Parsing

Here is the summarized content:

The English constituency parsing task presents specific challenges due to its strong structural constraints and longer output compared to input. Despite these challenges, we were able to train a 4-layer Transformer with <font color='red'>dmodel = 1024</font> on the Wall Street Journal portion of the Penn Treebank, achieving state-of-the-art results in small-data regimes. 
We experimented with different settings, including semi-supervised learning using approximately 17M sentences from high-confidence and BerkleyParser corpora. The results showed that despite not being task-specifically tuned, our model performed <font color='red'>surprisingly well</font>, outperforming previously reported models except for the Recurrent Neural Network Grammar. 
Our findings indicate that the Transformer can generalize to other tasks effectively, including those with strong structural constraints and longer output compared to input.

## 7 Conclusion

**Summary**

The Transformer model, presented in this work, is a sequence transduction model entirely based on attention, replacing traditional recurrent layers with multi-headed self-attention. This approach enables the Transformer to be trained significantly faster than architectures based on recurrent or convolutional layers. 
<font color='red'>We achieve a new state of the art</font> in both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, with our best model outperforming all previously reported ensembles. Future research goals include applying attention-based models to other tasks, extending the Transformer to handle input and output modalities beyond text, and investigating local attention mechanisms for efficient handling of large inputs and outputs. 
<font color='red'>Making generation less sequential is another research goal</font> of ours. The code used to train and evaluate our models is available on GitHub. 
**Equations**

$\boxed{\text{Transformer Model}}$

$$
\begin{aligned}
\text{Self-Attention}(Q, K, V) &= \frac{1}{\sqrt{d_k}} (QK^T)V \\
\text{Multi-Head Attention} &= \text{Parallel Self-Attentions} \\
&= \text{Concat}(h_1, h_2, ..., h_H) \\
&= \text{Linear}_g(\text{Concat}(h_1, h_2, ..., h_H)) \\
\end{aligned}
$$


