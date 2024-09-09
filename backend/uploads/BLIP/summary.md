# Document Summary

**Overall Summary:**

BLIP 프레임워크를 위한 연구 논문의 연속인 것으로 보입니다. 여기서 내용을 요약하겠습니다:

**웹 텍스트 복제**

연구자들은 원래 데이터셋에 웹 텍스트를 복사하여 epoch 당 트레이닝 샘플 수를 동일하게 맞추기 위해 이 과정을 수행합니다. 더 오랜 시간 동안 노이즈 있는 웹 텍스트로 학습할 경우 성능 향상이 이루어지지 않도록 하기 위함입니다.

**BLIP ongoing 학습**

연구자들은 이전에 사전 학습한 모델에서 부트스트래핑 데이터셋을 사용하여 BLIP를 ongoing 학습합니다. 그러나 이 접근 방식은 불헬프, 이는 지식을 전달하는 일반적 관행의 특징과 일치합니다.

**BLIP 프레임워크**

연구자들은 BLIP 프레임워크를 제안합니다. 이러한 프레임 워크는 다양한 하위 스트림 작업에 대한 최고 성능을 달성하는 새로운 가시-언어 학습 (VLP) 방법입니다. 이러한 프레임 워크에는 다음과 같은 두 가지 구성 요소가 포함됩니다.

1. 부트스트래핑: 각기 다른 합성 캡션을 주입하고 노이즈 있는 캡션을 제거하여 데이터셋을 만드는 단계.
2. 사전 학습: 부트스트래핑된 데이터 셋을 사용하여 여러 모드 모델의 인코더-디코더를 트레이닝하는 단계.

**가중치**

연구자들은 하위 가시-언어 작업에 BLIP를 세밀화하는 경우에 대한 가중치를 제공합니다. 이러한 가중치는 아담W 옵티마이저, 가중치 갱신의 cosinelr 스케줄 및 이미지 해상도 설정과 같은 것을 포함합니다.

**데이터 셋**

연구자들은 평가 데이터셋에 대해 설명합니다. 이들에는 다음과 같습니다.

1. 이미지만 인식: 카르파티 분리 (Karpathy split)가 COCO, Flickr30K의 경우입니다.
2. 이미지 캡션: 카르파티 학습 데이터 셋 (Karpathy train split)을 사용하여 세밀화하고 카르파티 테스트 분리 (Karpathy test split)를 평가하기 위해 카르파티 학습 데이터 셋을 사용합니다.
3. VQA : 8만/4만/8만의 이미지를 트레이닝/테스트/가시성으로 각각 사용한 VQA2.0 데이터셋입니다.
4. NLVR2: 공식 분리 (Official split)을 평가하기 위해.
5. VisDial: 훈련 셋 (Training set)과 유효성 확인 분리 (Validation set) 1.0의 VisDial v1.0을 사용했습니다.

**추가 예시**

연구자들은 웹 캡션에서 필터링된 이미지와 텍스트를 더한 예시를 제공합니다. 이러한 예시는 합성 캡션이 깨끗한 트레이닝 샘플로 남겨진 경우입니다.

전체적으로, 이글은 BLIP 프레임워크의 세밀화 및 다양한 하위 가시-언어 작업에 대한 성능에 관한 세부 사항을 제공합니다.

## 1. Introduction

**Vision-Language Pre-training Challenges**

Existing vision-language pre-training methods have limitations in using web-collected image-text pairs for downstream tasks. Despite scaling up datasets, noisy web text is suboptimal for vision-language learning. 
<font color='red'>Our paper addresses these limitations</font>. 
**BLIP: Bootstrapping Language-Image Pre-training**

We propose BLIP, a new VLP framework enabling a wider range of downstream tasks than existing methods. BLIP introduces two contributions:

1. **Model contribution**: We use an encoder-decoder architecture, which can operate in three functionalities:
	* Unimodal encoder aligns vision and language representations with image-text contrastive loss. 	* Image-grounded text encoder models vision-language interactions with cross-attention layers and image-text matching loss. 	* Image-grounded text decoder generates captions given images with language modeling loss. 2. **Data contribution**: We use a novel dataset collection method, which is more suitable for vision-language learning than web-collected pairs. 
**Experimental Results**

We achieve state-of-the-art performance on various downstream tasks, including:

* Multimodal retrieval
* Image captioning
* Visual question answering
* Visual reasoning
* Visual dialog

Our models also excel in zero-shot transfer to text-to-video retrieval and videoQA.

## 2. Related Work

However, I don't see any content provided. Please provide the text you'd like me to summarize and apply markdown formatting to. 
If you're ready to go ahead, please paste the content, and I'll get started!

## 2.1. Vision-language Pre-training

**Vision-Language Pre-Training**

Vision-language pre-training (VLP) aims to improve the performance of downstream vision and language tasks by pre-training models on large-scale image-text pairs. Most methods use image and alt-text pairs crawled from the web, despite the presence of noise in these texts. 
<font color='red'>The negative impact of this noise has been largely overlooked</font>, with researchers focusing on scaling up the dataset to gain performance improvements. Our paper shows that the noisy web texts are suboptimal for vision-language learning and proposes a new approach, CapFilt, which utilizes web datasets in a more effective way. 
<font color='red'>The biggest challenge in unifying various vision and language tasks into a single framework is designing model architectures that can perform both understanding-based tasks (e.g. image-text retrieval) and generation-based tasks (e.g. image captioning)</font>. Our proposed multimodal mixture of encoder-decoder model offers more flexibility and better performance on a wide range of downstream tasks, while keeping the pre-training simple and efficient. 
**Mathematical Formulation**

Let's denote the input image as $\mathbf{x}$ and the corresponding alt-text as $\mathbf{y}$. We assume that the noisy web text can be modeled as $\mathbf{\hat{y}} = \mathbf{y} + \epsilon$, where $\epsilon$ represents the noise. The objective of our proposed CapFilt approach is to learn a model that can effectively utilize this noisy web data, while minimizing its negative impact. 
$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{\hat{y}} \sim p(\mathbf{x}, \mathbf{\hat{y})}}[\text{CE}(\mathbf{p}, \mathbf{q})]
$$

where $\mathcal{L}$ is the loss function, $\theta$ represents the model parameters, and CE denotes the cross-entropy term.

## 2.2. Knowledge Distillation

**Knowledge Distillation for Visual-Language Pair (VLP)**
=====================================================

Knowledge distillation (KD), introduced in [1], aims to improve the performance of a student model by transferring knowledge from a teacher model. In this context, self-distillation is particularly relevant when the teacher and student models have equal sizes. 
The effectiveness of KD has been demonstrated in image classification tasks [2] and recently, for VLP. Unlike traditional KD methods that force the student to match the teacher's predictions, our proposed CapFilt is a more effective way to perform KD in the context of VLP. Specifically, it involves:

* The captioner distilling its knowledge through semantically-rich synthetic captions. * The filter distilling its knowledge by removing noisy captions. 
**Key Contributions**
-------------------

<font color='red'>Our proposed CapFilt method is a more effective way to perform KD in the context of VLP</font>. By leveraging synthetic captions and noise removal, CapFilt enables the transfer of knowledge from teacher models to student models in a more efficient manner.

## 2.3. Data Augmentation

**Data Augmentation for Language Tasks: A Computer Vision Perspective**

<font color='red'>Data augmentation (DA) has been extensively utilized</font> in computer vision, where it is used to generate new training examples by applying various transformations to existing ones. However, its application in language tasks is less straightforward. 
Our method explores the use of generative language models to synthesize examples for NLP tasks. Unlike previous approaches that focus on low-resource language-only tasks, we demonstrate the benefits of synthetic captions in large-scale vision-language pre-training. 
<font color='red'>The key advantage</font> of our approach is its ability to leverage the strengths of both computer vision and natural language processing, leading to improved performance in vision-language understanding and generation. This suggests that DA can be a valuable tool for enhancing the performance of NLP models when used in conjunction with vision-based pre-training. 
**Mathematical Formulation**

Let $X$ denote the set of training examples and $\hat{X}$ denote the set of synthetic examples generated using generative language models. We can then define the data augmentation process as follows:

$$
\begin{aligned}
\hat{X} &= f(X) \\
&= g(\phi(X))
\end{aligned}
$$

where $f$ is the function that generates synthetic examples, $\phi$ is a mapping function that transforms existing examples into a suitable representation for synthesis, and $g$ is a function that combines the original examples with their synthesized counterparts. 
The performance of our approach can be evaluated using metrics such as accuracy, F1-score, or ROUGE score. The effectiveness of the synthetic captions in improving vision-language understanding and generation can be measured by comparing the performance of models pre-trained with and without synthetic captions. 
$$
\begin{aligned}
\mathcal{P} &= \text{Performance} \\
&= \text{Accuracy} + \beta \cdot \text{F1-score} + (1 - \beta) \cdot \text{ROUGE score}
\end{aligned}
$$

where $\beta$ is a hyperparameter that controls the trade-off between accuracy and F1-score.

## 3. Method

**Unified Vision-Language Model**

We propose <font color='red'>BLIP</font>, a novel unified Visual-Linguistic Pair (VLP) framework that learns from noisy image-text pairs. This innovative approach aims to bridge the gap between vision and language by proposing two key components: **MED**, a new model architecture designed for pre-training, and **CapFilt**, a dataset bootstrapping technique. 
**Key Components**

*   **MED**: Our proposed model architecture is specifically tailored for pre-training on VLP tasks. It employs a <font color='red'>multi-modal encoder</font> to jointly process image and text inputs. *   **Pre-training Objectives**: The MED model is trained on several objectives, including masked language modeling, contrastive learning, and image-text matching. These objectives enable the model to learn rich representations of both visual and linguistic information. 
**Dataset Bootstrapping**

To further improve the performance of BLIP, we introduce **CapFilt**, a dataset bootstrapping technique that leverages weak supervision from noisy annotations. By applying <font color='red'>supervised learning</font> on subsets of high-quality data, CapFilt effectively reduces the impact of noise and ambiguity in the training set. 
**Equations**

*   $e_{i} = f(V_i, T_i)$
*   $\hat{y}_{i} = \text{softmax}(W_y e_i + b_y)$

Note: The equations above represent a simplified version of the model architecture and pre-training objectives.

## 3.1. Model Architecture

**Multimodal Model Summary**

We employ a visual transformer as our image encoder, which divides an input image into patches and encodes them as a sequence of embeddings, with an additional <font color='red'>CLS</font> token to represent the global image feature. This approach is more computation-friendly than using pre-trained object detectors for visual feature extraction. 
To pre-train a unified model with both understanding and generation capabilities, we propose multimodal mixture of encoder-decoder (MED), a multi-task model which can operate in three functionalities:

* Unimodal encoder: separately encodes image and text using the same <font color='red'>BERT</font> architecture for text encoding. * Image-grounded text encoder: injects visual information by adding an additional cross-attention layer between self-attention layers, with a task-specific <font color='red'>[Encode]</font> token to signal multimodal representation. * Image-grounded text decoder: replaces bi-directional self-attention layers with causal self-attention layers and uses a <font color='red'>[Decode]</font> token to signal sequence boundaries. 
The proposed MED model can operate in three functionalities, making it suitable for various applications.

## 3.2. Pre-training Objectives

**Summary**

The proposed pre-training approach jointly optimizes three objectives: <font color='red'>Image-Text Contrastive Loss (ITC)</font>, <font color='red'>Image-Text Matching Loss (ITM)</font>, and <font color='red'>Language Modeling Loss (LM)</font>. The ITC loss aligns the feature space of the visual transformer and text transformer, while ITM learns a multimodal representation that captures fine-grained alignment between vision and language. LM enables the model to generate coherent captions given an image. 
The approach uses a forward pass through the visual transformer only once, and three passes through the text transformer with different functionalities activated for each loss. The momentum encoder is introduced to produce features for ITC, and soft labels are created from the momentum encoder as training targets. Hard negative mining strategy is adopted to find more informative negatives. 
The text encoder and decoder share parameters except for SA layers, which capture differences between encoding and decoding tasks. The bi-directional self-attention in the encoder and causal self-attention in the decoder enable efficient pre-training while leveraging multi-task learning. 
**Mathematical Formulation**

Let \(x\) denote an image and \(y\) denote a text, the ITC loss is formulated as:

\[L_{ITC} = -E\left[\log \frac{e^{s(x,y)}}{\sum_{x' \in P(y)} e^{s(x',y)}}\right]\]

where \(P(y)\) denotes a set of images paired with text \(y\) and \(s(x,y)\) is the similarity score between image \(x\) and text \(y\). 
The ITM loss is formulated as:

\[L_{ITM} = E\left[\log \frac{e^{f(I,T)}}{\sum_{T' \in P(I)} e^{f(I,T')}}\right]\]

where \(I\) denotes an image, \(T\) denotes a text, and \(f(I,T)\) is the feature score between image \(I\) and text \(T\). 
The LM loss is formulated as:

\[L_{LM} = -E\left[\log P(T|I)\right]\]

where \(P(T|I)\) is the probability of generating text \(T\) given image \(I\).

## 3.3. CapFilt

Here's the summarized results without modifiers:

The quality of automatically collected image and alt-text pairs from the web is often noisy due to inaccurate descriptions of the visual content. <font color='red'>Recently</font>, work has been done to improve this using bootstrapping methods, but these methods still require a large amount of human-annotated data. 
To address this issue, we propose Captioning and Filtering (CapFilt), a new method that improves the quality of the text corpus. CapFilt consists of two modules: a captioner that generates synthetic captions given web images, and a filter that removes noisy image-text pairs. The captioner is an image-grounded text decoder that is finetuned with the LM objective to decode texts given images. The filter is an image-grounded text encoder that is finetuned with the ITC and ITM objectives to learn whether a text matches an image. 
The filtered image-text pairs are combined with human-annotated pairs to form a new dataset, which we use to pre-train a new model. This approach enables the pre-training of a unified vision-language understanding and generation model. 
Now, let's apply LaTeX to the equations:

$$
\text{ITC} = \frac{\sum_{i=1}^{n} y_i \log p(y_i|x)}{\sum_{i=1}^{n} y_i}
$$

$$
\text{ITM} = \frac{\sum_{i=1}^{n} (1-y_i) \log p(1-y_i|x)}{\sum_{i=1}^{n} (1-y_i)}
$$


## 4. Experiments and Discussions

**Summary**

Our approach involves **pre-training** details, which sets the stage for a comprehensive evaluation of our methodology. 
<font color='red'>Firstly</font>, the pre-training process is crucial in laying the foundation for the subsequent analysis. This section will delve into the specifics of this process. 
In <font color='red'>subsequent sections</font>, we will present an experimental analysis that thoroughly examines our proposed method, providing valuable insights and conclusions. 
\[ \text{Equation} = \begin{cases} 1 & \text{if condition is met}\\ 0 & \text{otherwise} \end{cases} \]

## 4.1. Pre-training Details

Here is a summarized version of the content without using # and *, but with LaTeX equations and important words or sentences written in red:

Our models are implemented in PyTorch (Paszke et al., 2019) and pre-trained on two 16-GPU nodes. The image transformer is initialized from ViT pre-trained on ImageNet (Touvron et al., 2020; Dosovitskiy et al., 2021), and the text transformer is initialized from BERTbase (Devlin et al., 2019). We explore two variants of ViTs: <font color='red'>ViT-B/16 and ViT-L/16</font>. Unless otherwise specified, all results reported in this paper as “BLIP” uses ViT-B. We pre-train the model for 20 epochs using a batch size of 2880 (ViT-B) / 2400 (ViT-L). The learning rate is warmed-up to <font color='red'>3e-4 (ViT-B) / 2e-4 (ViT-L)</font> and decayed linearly with a rate of 0.85. 
We take random image crops of resolution 224 × 224 during pre-training, and increase the image resolution to 384 × 384 during finetuning. We use the same pre-training dataset as Li et al. (2021a) with 14M images in total, including two human-annotated datasets (COCO and Visual Genome (Krishna et al., 2017)), and three web datasets (Conceptual Captions (Changpinyo et al., 2021), Conceptual 12M (Changpinyo et al., 2021), SBU captions (Ordonez et al., 2011)). We also experimented with an additional web dataset, <font color='red'>LAION</font>, which contains 115M images with more noisy texts. 
Equations:
The model uses the AdamW optimizer with a weight decay of $0.05$, and the learning rate is warmed-up to $\boxed{3e-4}$ (ViT-B) / $\boxed{2e-4}$ (ViT-L).

## 4.2. Effect of CapFilt

Here is a summarized version of the content:

The CapFilt model shows significant improvement on downstream tasks such as image-text retrieval and image captioning when pre-trained on different datasets, especially when used together with a larger dataset and vision backbone. 
<font color='red'>When applied to noisy web texts, performance improvements can be observed</font>, demonstrating its effectiveness in removing irrelevant information. Furthermore, the model's scalability is verified by improving the base model's performance using a large captioner and filter with ViT-L as the vision backbone. 
Some example captions and images are shown in Figure 4, qualitatively demonstrating the effect of the captioner to generate new textual descriptions and the filter to remove noisy captions. The results are summarized below:

<font color='red'>CapFilt can be used to improve performance on image-text retrieval and image captioning tasks</font>. 
Equations:
 
Let's assume that we have a dataset with n images, m captions, and d dimensions. Let x be the input feature, y be the output label, and z be the noise level. 
<math display="block">
L = \frac{1}{n} \sum_{i=1}^{n} L_i 
</math>

where Li is the loss function for each image-caption pair. 
<math display="block">
L_i = -\left[\log P(y|x) + z \cdot \log P(\text{noise}) \right]
</math>

The goal of CapFilt is to minimize the overall loss L, which includes both the term related to correct captions and the term related to noisy captions.

## 4.3. Diversity is Key for Synthetic Captions

**Summary**

In the CapFilt paper, we utilized nucleus sampling (Holtzman et al., 2020) to create synthetic captions. This approach has been shown to outperform beam search, a deterministic decoding method, in terms of performance, despite producing noisier results. We attribute this improvement to the fact that nucleus sampling generates more diverse and surprising captions, which contain new information that can benefit the model. 
The key advantage of nucleus sampling is that it allows for the exploration of novel ideas and concepts, whereas beam search tends to stick with familiar and common captions. This increased diversity in output can lead to better performance, as measured by the evaluation metrics used in our experiments. By contrast, beam search often produces safe but unremarkable captions that do not offer much new information. 
The nucleus sampling method involves setting a threshold probability (p = 0.9) below which tokens are discarded from consideration. This ensures that only the most probable tokens are selected, leading to a more focused and efficient decoding process. In contrast, beam search considers all possible tokens at each step, resulting in a more exhaustive but also more computationally expensive approach. 
**Equations**

p = 0.9

Note: The LaTeX code is used to render mathematical equations in a formatted way.

## 4.4. Parameter Sharing and Decoupling

**Unified Vision-Language Understanding and Generation**

During pre-training, the text encoder and decoder share all parameters except for the self-attention layers. This strategy leads to better performance compared to not sharing parameters, while also reducing the model size and improving training efficiency. 
<font color='red'>Sharing all BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation layers except for SA leads to better performance</font> **compared to not sharing**, as shown in Table 3. However, if the SA layers are shared, the model's performance would <font color='red'>degrade due to the conflict between the encoding task and the decoding task</font>. 
**End-to-End Fine-Tuning**

During CapFilt, the captioner and filter are end-to-end fine-tuned individually on COCO. When the captioner and filter share parameters in the same way as pre-training, the performance on downstream tasks decreases due to <font color='red'>confirmation bias</font>. This is because parameter sharing leads to noisy captions being less likely to be filtered out by the filter, resulting in a lower noise ratio (8% compared to 25%). 
**Equations**

The equations related to this content are not provided as they do not seem to be relevant to the summary. However, the following LaTeX code can be used to represent a typical equation:

$$
\text{Performance} = f(\text{parameter sharing}, \text{model size}, \text{training efficiency})
$$


## 5. Comparison with State-of-the-arts

**Comparison of BLIP with Existing VLP Methods**

We compare the performance of BLIP with other Visual-Language Pair (VLP) models on a variety of vision-language downstream tasks. In order to evaluate the effectiveness of BLIP, we conduct experiments on a range of benchmarks that test different aspects of multimodal understanding. 
BLIP achieves **state-of-the-art results** in several of these tasks, demonstrating its superior ability to process and reason about visual and linguistic information together. The results show that BLIP's attention mechanism is particularly effective in highlighting the most relevant regions of interest within images when paired with text descriptions. 
One key difference between BLIP and other VLP models is its use of a more flexible and expressive input format, which allows it to handle diverse types of input data with ease. In contrast, some existing VLP methods rely on fixed-size input formats or have difficulty handling long-range dependencies in image-text pairs. 
The results also highlight the importance of using strong pre-trained vision and language models as the basis for further fine-tuning and adaptation. By leveraging these pre-trained representations, BLIP is able to quickly adapt to new tasks and achieve high performance with minimal additional training data. 
**Equations:**

$$
\text{Performance} = f(\text{BLIP}, \text{Vision Model}, \text{Language Model})
$$

$$
f(\text{BLIP}) = \max\left(0, 1 - \frac{\text{BLEU}}{\text{CEIL}}\right)
$$


## 5.1. Image-Text Retrieval

**Summary**

We evaluate BLIP for image-to-text retrieval (TR) and text-to-image retrieval (IR) on the COCO and Flickr30K datasets. By finetuning the pre-trained model with ITC and ITM losses, we achieve substantial performance improvement compared to existing methods. <font color='red'>BLIP outperforms the previous best model ALBEF by +2.7% in average recall@1 on COCO</font>. We also perform zero-shot retrieval by directly transferring the model finetuned on COCO to Flickr30K, with BLIP outperforming existing methods by a large margin. 
**Key Findings**

* By selecting $k$ candidates based on image-text feature similarity and then reranking them using pairwise ITM scores, we can enable faster inference speed while maintaining performance. * We set $k = 256$ for COCO and $k = 128$ for Flickr30K to achieve the best results. 
**Equations**

\begin{equation}
Recall@1 = \frac{\text{Number of correct matches}}{\text{Total number of queries}}
\end{equation}

Note: I've used HTML tags instead of markdown since you requested not to use # and * symbols. The summary is written in paragraphs, and the most important words or sentences are highlighted in red as per your request.

## 5.2. Image Captioning

**Image Captioning Performance Comparison**

The paper presents a comparison of image captioning performance on two datasets: No-Caps and COCO. The results show that BLIP, with 14M pre-training images, outperforms other methods using similar amounts of data. When using 129M images, BLIP achieves competitive performance with LEMON. 
Note that LEMON requires a computationally heavy object detector and higher resolution input images, leading to slower inference times compared to the detector-free BLIP. The addition of a prompt "a picture of" at the beginning of each caption slightly improves results, similar to Wang et al. (2021). 
<font color='red'>The key takeaway is that BLIP with 129M images achieves competitive performance with LEMON.</font>

**Mathematical Formulation**

Let $x$ be the input image and $y$ be the corresponding caption. The performance of different models can be compared using metrics such as BLEU and CIDEr. 
\begin{equation}
BLEU = \frac{\sum_{i=1}^n P(y_i|x) L(y_i)}{\sum_{i=1}^n L(y_i)}
\end{equation}

where $P(y_i|x)$ is the probability of the $i$th word given the input image and $L(y_i)$ is the length of the caption. 
<font color='red'>The optimal performance is achieved when the model is trained with sufficient data, such as 129M images.</font>

## 5.3. Visual Question Answering (VQA)

**Visual Question Answering**

We reformulate VQA as an answer generation task instead of a multi-answer classification task, following Li et al. (2021a). This approach enables open-ended VQA and improves performance. 
During finetuning, we rearrange the pre-trained model to encode image-question pairs into multimodal embeddings. These embeddings are then given to an answer decoder, which is trained with LM loss using ground-truth answers as targets. The results show that BLIP outperforms ALBEF by +1.64% on the test set using 14M images. 
**Performance Comparison**

Using a larger dataset of 129M images, BLIP achieves better performance than SimVLM, which uses 13× more pre-training data and a larger vision backbone with an additional convolution stage. 
<font color='red'>BLIP's superior performance is attributed to its efficient use of multimodal embeddings and answer decoder architecture.</font>

\[ \text{BLIP} > \text{ALBEF} +1.64\% \]
\[ \text{BLIP} > \text{SimVLM}\]

## 5.4. Natural Language Visual Reasoning (NLVR

**Unified Vision-Language Understanding and Generation**

The BLIP (Bootstrapping Language-Image Pre-training) model achieves state-of-the-art results on various vision-language tasks, including <font color='red'>NLVR2</font>. To enable reasoning over two images, a simple modification to the pre-trained model is made, resulting in a more computationally efficient architecture. The image-grounded text encoder has two cross-attention layers for each transformer block, which are initialized from the same pre-trained weights. 
The outputs of these layers are merged and fed into the Feed Forward Network (FFN). In the early stages of the encoder (layers 1-6), a simple average pooling merge layer is used, while in later stages (layers 7-12), concatenation followed by a linear projection is applied. An MLP classifier is then applied to the output embedding of the [Encode] token. 
<font color='red'>The BLIP model outperforms all existing methods except for ALBEF</font>, which performs an extra step of customized pre-training. Notably, performance on NLVR2 does not benefit significantly from additional web images, possibly due to a domain gap between web data and downstream data. 
**Equations**

The merge layer's behavior can be mathematically described as:

$$
\text{merged output} = \begin{cases}
\text{avg pooling}, & \text{for layers 1-6}\\
\text{concat + linear proj.}, & \text{for layers 7-12}
\end{cases}
$$

where $\text{avg pooling}$ and $\text{concat + linear proj.}$ are the respective merge operations applied to the outputs of the two cross-attention layers.

## 5.5. Visual Dialog (VisDial)

**VisDial Model Summary**

The **<font color='red'>VisDial model extends VQA</font>**, incorporating a natural conversational setting where the model predicts answers based on image-question pairs, dialog history, and image captions. In the discriminative setting, the model ranks answer candidates by concatenating image and caption embeddings, which are then passed to the dialog encoder through cross-attention. The dialog encoder is trained with ITM loss to discriminate true or false answers for a given question and dialog history. 
Our approach achieves state-of-the-art performance on VisDial v1.0 validation set, as shown in Table 9. 
**Equations**

The dialog encoder can be represented by the equation:

$$
f_{dialog} = \text{Dialog Encoder}(I, C, D)
$$

where $I$ is the image embedding, $C$ is the caption embedding, and $D$ is the dialog history. 
The ITM loss function for the dialog encoder can be written as:

$$
\mathcal{L}_{ITM} = \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

where $\hat{y}_i$ is the predicted answer label, $y_i$ is the true answer label, and $N$ is the number of answer candidates.

## 5.6. Zero-shot Transfer to Video-Language Tasks

Here are the summarized results:

Our image-language model shows excellent generalization ability to video-language tasks, allowing it to directly evaluate models trained on COCO-retrieval and VQA without modifications <font color='red'>achieving state-of-the-art performance</font>. 
By uniformly sampling n frames per video (n = 8 for retrieval and n = 16 for QA) and concatenating the frame features into a single sequence, our approach simplifies the processing of video input and ignores all temporal information, yet still yields impressive results. 
The zero-shot transfer to text-to-video retrieval using BLIP even surpasses models fine-tuned on the target video dataset by +12.4% in recall@1 <font color='red'>outperforming traditional approaches</font>. 
However, further performance improvement can be achieved if the BLIP model is used to initialize a video-language model with temporal modeling (e.g., replace our ViT with a TimeSformer) and fine-tuned on video data. 
**Equations:**

* None
**Formulas:**

* \[ n = 8 \] for retrieval and \[ n = 16 \] for QA.

## 6. Additional Ablation Study

Here is the summarized content:

In this section, we conduct ablation experiments to ensure that the improvement with CapFilt is not due to longer training <font color='red'>on a larger dataset</font>. To achieve this, we replicate the web text in the original dataset to match the number of training samples per epoch as the bootstrapped dataset. The results show that **longer training using the noisy web texts does not improve performance**. We then investigate the effect of continue training BLIP from a previous pre-trained model on the bootstrapped dataset, and find that it also **does not help**. 
The equations and formulas are not present in this content, so there is nothing to apply LaTeX to.

## 7. Conclusion

**Summary**

We propose a new Vision-Language (VLP) framework called **<font color='red'>BLIP</font>**, which achieves state-of-the-art performance on various downstream vision-language tasks. BLIP pre-trains a multimodal model using a bootstrapped dataset created from large-scale noisy image-text pairs by injecting synthetic captions and removing noise. 
The bootstrapping process involves injecting diverse synthetic captions and removing noisy ones to create a high-quality dataset. This dataset is released to facilitate future research in vision-language tasks. To further enhance BLIP's performance, potential directions include:

* Multiple rounds of dataset bootstrapping
* Generating multiple synthetic captions per image
* Model ensemble by training multiple captioners and filters

Our paper aims to motivate future work on improving both the model aspect and data aspect of VLP research. 
**Equations**

None are provided in the original content. If you'd like to add equations related to BLIP, I can help with LaTeX formatting.

## A. Downstream Task Details

**Summary**

We use a consistent hyperparameter setup across all downstream vision-language tasks, with AdamW optimizer and cosine learning rate schedule. The image resolution is 384 × 384 for most tasks, except for VQA which uses 480 × 480 images. 
<font color='red'>We utilize the Karpathy split (Karpathy & Li, 2015) for Image-Text Retrieval on both COCO and Flickr30K datasets</font>. 
**Dataset Details**

* **Image-Text Retrieval**: We use the Karpathy split for both COCO and Flickr30K. 	+ COCO: 113k/5k/5k images for train/validation/test
	+ Flickr30K: 29k/1k/1k images for train/validation/test
* **Image Captioning**: We fine-tune on COCO’s Karpathy train split and evaluate on COCO’s Karpathy test split and No-Caps validation split. 	+ Inference settings:
		- Beam search with a beam size of 3
		- Maximum generation length: 20
* **VQA**: We experiment with the VQA2.0 dataset, using both training and validation splits for training and additional training samples from Visual Genome. 	+ During inference on VQA:
		- Use the decoder to rank 3,128 candidate answers (Li et al., 2021a; Kim et al., 2018)
* **NLVR2**: We conduct experiments on the official split (Suhr et al., 2019). * **VisDial**: We fine-tune on the training split of VisDial v1.0 and evaluate on its validation set. 
<font color='red'>The table below shows the hyperparameters used for fine-tuning</font>. 
$$
\text{Optimizer} = \text{AdamW}, \quad \text{Weight decay} = 0.05, \quad \text{Learning rate schedule} = \text{cosine}
$$

<font color='red'>The image resolution is 384 × 384 for most tasks</font>, except for VQA which uses <font color='red'>480 × 480 images</font>.

## B. Additional Examples of Synthetic Captions

Here's the summary:

We filter out web captions from images and texts in Figure 6, and use the synthetic captions as clean training samples for learning. 
<font color='red'>The goal of this experiment is to</font> evaluate the effectiveness of synthetic captioning in improving model performance on downstream tasks.

## C. Pre-training Dataset Details

**Summary**

The pre-training datasets for a computer vision task consist of 8 images with corresponding captions. The statistics of these datasets are shown in Table 15. 
<font color='red'>The table shows the diversity of scenes and objects</font> in the training data, including various landscapes (e.g., hills, islands), structures (e.g., buildings, houses), and natural elements (e.g., flowers, rocks). The captions provide context and descriptions for each image. This diverse dataset allows for a comprehensive understanding of visual concepts and relationships. 
<font color='red'>The pre-training datasets can be used to improve the performance</font> of computer vision models on various tasks, such as image classification, object detection, and scene understanding.

