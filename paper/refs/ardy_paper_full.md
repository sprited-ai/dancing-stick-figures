ARDY: Autoregressive Diffusion with Hybrid Representation for
Interactive Human Motion Generation
KAIFENG ZHAO, NVIDIA, Switzerland and ETH Zürich, Switzerland
MATHIS PETROVICH, NVIDIA, Switzerland
HAOTIAN ZHANG, NVIDIA, USA
TINGWU WANG, NVIDIA, USA
SIYU TANG, ETH Zürich, Switzerland
DAVIS REMPE, NVIDIA, USA
                                                                                                                                                   Root velocity control via keyboard




                                                                                                                     End-effector
                                                                                                                     joints

                          Root
                          trajectory



                                                    Full body
                  Root                              keyframe
                  waypoint




   A happy person, leaning to the left, jogs in
    a fast circular arc to their left, then stop.
                                                           A person is stealthily walking
                                                        sideways to their left at a slow pace.
                                                                                                 A person opens a door and walks out
                                                                                                  the door before closing it behind.
                                                                                                                                        ……                                     ! 33ms
Fig. 1. We present ARDY, an autoregressive diffusion model designed for interactive human motion generation. Our approach natively supports online text
prompting alongside a comprehensive suite of flexible kinematic constraints — including root waypoints and trajectories, full-body keyframes, and sparse
joint positions and rotations — over long horizons. ARDY enables controllable and responsive interactive motion synthesis from real-time user inputs such as
mouse and keyboard commands, with our efficient 4-step diffusion model achieving an average generation latency of 33 ms.

Generating realistic 3D human motions in real-time within interactive ap-                                  propose a two-stage autoregressive transformer denoiser that features vari-
plications is key for animation, simulation, and humanoid robotics. While                                  able history context and supports conditioning on flexible, long-horizon
recent offline motion generation approaches offer precise control via text and                             kinematic constraints. By training on a large-scale motion capture dataset
kinematic constraints, they lack the inference speed required for interactive                              and being directly conditioned on text labels and kinematic constraints
settings. Conversely, existing online methods enable real-time synthesis                                   sampled from ground truth poses, ARDY natively learns controllable gen-
but often sacrifice controllability or struggle with complex text semantics                                eration that supports online prompting and flexible long-horizon goals.
and long-horizon goals due to limited context windows. In this work, we                                    Extensive evaluations on the HumanML3D benchmark and the large-scale,
introduce ARDY, a streaming generation framework that bridges this gap                                     high-fidelity Bones Rigplay dataset demonstrate ARDY’s high motion quality
by enabling high-fidelity motion generation controllable via online text                                   and constraint adherence, validating the efficacy of our key architectural
prompts and flexible kinematic constraints. ARDY employs a hybrid repre-                                   decisions. Finally, we demonstrate the method’s practical versatility through
sentation that combines explicit root features with a latent body embedding,                               an interactive demo featuring dynamic text control, diverse keyframe pose
balancing precise trajectory control with efficient generative learning. We                                constraints, path following, and interactive locomotion control via mouse
                                                                                                           and keyboard. Supplementary video results, code, and model releases can be
Authors’ Contact Information: Kaifeng Zhao, kaifeng.zhao@inf.ethz.ch, NVIDIA,                              found at https://research.nvidia.com/labs/sil/projects/ardy/.
Switzerland and ETH Zürich, Switzerland; Mathis Petrovich, mpetrovich@nvidia.com,
NVIDIA, Switzerland; Haotian Zhang, haotianz@nvidia.com, NVIDIA, USA; Tingwu
Wang, tingwuw@nvidia.com, NVIDIA, USA; Siyu Tang, siyu.tang@inf.ethz.ch, ETH                               CCS Concepts: • Computing methodologies → Motion processing.
Zürich, Switzerland; Davis Rempe, drempe@nvidia.com, NVIDIA, USA.


                                                                                                           ACM Reference Format:
                                                                                                           Kaifeng Zhao, Mathis Petrovich, Haotian Zhang, Tingwu Wang, Siyu Tang,
This work is licensed under a Creative Commons Attribution 4.0 International License.
© 2026 Copyright held by the owner/author(s).
                                                                                                           and Davis Rempe. 2026. ARDY: Autoregressive Diffusion with Hybrid Rep-
ACM 1557-7368/2026/7-ART86                                                                                 resentation for Interactive Human Motion Generation. ACM Trans. Graph.
https://doi.org/10.1145/3811284                                                                            45, 4, Article 86 (July 2026), 14 pages. https://doi.org/10.1145/3811284


                                                                                                                       ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:2   •   Zhao et al.


1      Introduction                                                           complex long-term motion semantics and long-horizon kinematic
Learning to generate realistic 3D human motions has become a                  goal reaching. Moreover, the autoregressive denoiser employs an
promising direction with applications ranging from character an-              interleaved two-stage architecture: it first predicts the clean explicit
imation and simulation to humanoid robotics. Offline authoring                root, then predicts the clean latent body embedding conditioned
models can benefit animators and game developers through intu-                on the first-stage root prediction. These two stages operate in an
itive controls like text and kinematic constraints [Pinyoanuntapong           interleaved manner within the denoising loop, ensuring continuous
et al. 2025; Xie et al. 2024]. Meanwhile, interactive motion genera-          mutual influence between root and body motion. This staged de-
tors [Shi et al. 2024; Xiao et al. 2025] are key for characters in games      sign is crucial for simultaneously satisfying text instructions and
and simulations to react to their environment and user inputs in              kinematic constraints. By training on a large-scale dataset with text
real time. Besides digital humans, recent work in real-world hu-              labels and kinematic constraints sampled from the ground truth
manoid robot control [He et al. 2025; Liao et al. 2025; Luo et al. 2025;      motion itself, ARDY learns conditional generation that supports
Zhao et al. 2025b] relies heavily on high-quality human motions for           online prompting and long-horizon kinematic goals, eliminating
supervision during training or planning at runtime.                           the need for additional control modules [Pinyoanuntapong et al.
   Recent methods in offline motion modeling generate a full se-              2025; Shi et al. 2024; Zhao et al. 2025a] such as expensive test-time
quence of poses in parallel. Modern generative models such as                 optimization or RL-based control policies.
diffusion [Karunratanakul et al. 2023; Rempe et al. 2026; Tevet et al.           We present an interactive demo that highlights the practical ca-
2023; Zhang et al. 2024a] and generative masked modeling [Guo                 pabilities of our method, including dynamic text control, dense and
et al. 2024; Jiang et al. 2024a; Pinyoanuntapong et al. 2025] allow           sparse key-pose constraints, path following, and real-time locomo-
synthesized motions to follow complex text prompts and kinematic              tion control via mouse and keyboard. This demonstration showcases
constraints such as pose keyframes and joint positions. While these           the potential for generative models to power next-generation inter-
methods are expressive and controllable, their spatiotemporal design          active animation systems. Moreover, we validate our design choices
and/or slow inference time are usually not suitable for interactive           on the Bones Rigplay [Bones Studio 2026] dataset—featuring a signif-
applications such as computer games or robot control.                         icantly larger scale and higher quality than the public HumanML3D
   In contrast, online models generate motion at runtime [Chen et al.         dataset—to assess the impact of key architectural decisions. Fur-
2024; Holden et al. 2017; Ling et al. 2020], usually in an autoregres-        thermore, we evaluate ARDY against state-of-the-art offline and
sive fashion. While these models are fast and capable of producing            autoregressive conditional motion generation methods on the pub-
realistic animations, they tend to sacrifice controllability. Some ap-        lic HumanML3D [Guo et al. 2022] benchmark, validating its strong
proaches support text conditioning but lack kinematic control [Xiao           motion quality and kinematic constraint adherence in a controlled
et al. 2025], while others enable kinematic constraints but can not           setting that isolates the effects of proprietary data.
accept text input [Chen et al. 2024; Shi et al. 2024]. Although a few            In summary, the key contributions of this paper are (1) a hy-
recent methods integrate both text and kinematic constraints con-             brid latent-body explicit-root representation amenable to fast and
trol [Tevet et al. 2025; Zhao et al. 2025a], their restricted context         controllable motion generation, (2) a two-stage autoregressive dif-
windows limit the understanding of global text semantics and the              fusion model featuring variable history context length and support
execution of long-horizon kinematic goals.                                    for long-horizon kinematic constraint conditioning, including full-
   In this work, we aim to get the best of both: controllability through      body keyframes, root waypoints, root paths, and end-effector po-
complex text prompts and flexible kinematic goal constraints, while           sitions/rotations, and (3) an extensive evaluation on a large-scale,
generating motion in a streaming fashion that enables online in-              production-quality dataset that highlights the efficacy of our design
teractivity (see Fig. 1). To achieve this, we introduce ARDY, an              choices and demonstrates the strong capabilities of ARDY.
Auto-Regressive Diffusion model that leverages a hYbrid pose repre-
sentation to generate high-quality motion interactively, conditioned          2   Related Work
on online text prompts and flexible kinematic constraints from user           In this section, we summarize relevant work in conditional 3D
inputs. ARDY is comprised of two main components. First, ARDY                 human motion generation and how our method fits in context. For
employs a hybrid motion representation that decomposes motion                 this purpose, we define offline motion generation as a method that
into an explicit root feature and a latent body embedding derived             generates a full spatiotemporal sequence of poses in parallel, while
from a learned tokenizer. This hybrid representation enables explicit         online/interactive/runtime/streaming motion generation refers to an
and accurate root control during generation while maintaining a               autoregressive method that generates poses sequentially (either
compact representation for efficient generative learning. Second,             individually or in chunks) and can therefore react to dynamically
ARDY utilizes an autoregressive transformer denoiser for interactive          changing conditions (e.g., new text prompts or constraints).
motion generation, conditioned on a text prompt and kinematic con-
straints that can be spatiotemporally sparse and span long horizons.             Offline Human Motion Generation. A primary focus of many re-
To handle variable and potentially sparse constraints, we represent           cent offline motion generation works is text conditioning. Enabled
the constraints as a masked motion sequence that is injected as               by motion datasets with natural language descriptions [Plappert
input conditioning to the autoregressive denoiser. The denoiser               et al. 2016], early work on this problem employed VAE-based ar-
features a variable history context and supports kinematic goals ex-          chitectures for diverse generation [Guo et al. 2022; Petrovich et al.
tending beyond a single generation window, which are essential for            2022]. More recently, diffusion models have proven to be effective
                                                                              at capturing the complex distribution of text and motion, enabling

ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                                   ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation             •   86:3


Table 1. Method Feature Comparison. Comparison of the proposed ARDY with existing conditional 3D motion generation methods. We delineate various
capabilities including real-time performance, online prompting, supported spatial control types, the architectural mechanism of control (i.e., whether each
method requires test-time optimization or RL policies), and the maximum history and future context length in model generation.

                                             Real-time    Online text                      Spatial control                              Native control               Context length (s)
 Method
                                             generation   prompting
                                                                         Root trajectory     Joint position    Joint rotation   No optimization    No RL policy     History      Future
 MaskControl [Pinyoanuntapong et al. 2025]       ✗            ✗                ✓                   ✓                 ✗                ✗                   ✓          N/A         10.00
 Kimodo [Rempe et al. 2026]                      ✗            ✗                ✓                   ✓                 ✓                ✓                   ✓          N/A         10.00
 AMDM [Shi et al. 2024]                          ✓            ✗                ✓                   ✓                 ✗                ✓                   ✗          0.03        0.03
 CAMDM [Chen et al. 2024]                        ✓            ✗                ✓                   ✗                 ✗                ✓                   ✓          0.33        1.50
 MotionStreamer [Xiao et al. 2025]               ✓            ✓                ✗                   ✗                 ✗               N/A                 N/A         10.00       10.00
 DartControl [Zhao et al. 2025a]                 ✓            ✓                ✓                   ✓                 ✗                ✗                   ✗          0.07        0.27
 DiP [Tevet et al. 2025]                         ✓            ✓                ✓                   ✓                 ✗                ✓                   ✓          1.00        2.00
 ARDY (Ours)                                     ✓            ✓                ✓                   ✓                 ✓                ✓                   ✓          8.00        10.00



high-quality motion generation from prompts [Chen et al. 2023;                                   Matching [Holden et al. 2020] and Control Operators [Gou et al.
Tevet et al. 2023; Zhang et al. 2024a]. Motion diffusion models are                              2025] enable responsive real-time character control via learned sim-
also capable of flexible kinematic control, enabling “any-joint-any-                             ilarity metrics and modular control primitives rather than explicit
time” constraints on generated motions [Karunratanakul et al. 2024,                              generative modeling. Moving into generative approaches, autore-
2023; Rempe et al. 2026; Xie et al. 2024]. However, the iterative de-                            gressive VAE models learned a low-dimensional motion latent space
noising process for potentially long motions tends to be too slow for                            for task-based RL control [Ling et al. 2020; Zhang and Tang 2022] and
interactive applications. Some methods have considerably sped up                                 tracking via optimization [Rempe et al. 2021]. Similar approaches
the denoising process by reducing the number of required steps [Dai                              have learned human-object interactions [Hassan et al. 2021; Starke
et al. 2025; Zhou et al. 2024], but are still designed to generate all                           et al. 2019; Zhao et al. 2023] by conditioning the model on object
poses in parallel. While some diffusion approaches can handle a                                  geometry in addition to the future pose information.
temporal sequence of input prompts, these methods generate all                                      Autoregressive motion diffusion models have taken the approaches
prompts jointly offline [Barquero et al. 2024; Li et al. 2025; Petrovich                         developed for offline generation and made them amenable to in-
et al. 2024], which is not suitable for interactive applications.                                teractive settings, primarily through shorter motion generation
   Another line of work leverages a discrete tokenized representa-                               horizon and fewer denoising steps [Chen et al. 2024; Ji et al. 2025;
tion of human motion. Methods like MoMask [Guo et al. 2024] and                                  Jiang et al. 2024b; Shi et al. 2024; Wu et al. 2025; Zhang et al. 2025,
MMM [Pinyoanuntapong et al. 2024b] generate motion from text                                     2024b; Zhao et al. 2025a]. A-MDM learns to denoise the next pose
by training a VQ-VAE motion tokenizer followed by a masked trans-                                in a motion given the previous pose, and allows flexible kinematic
former that iteratively predicts masked poses, eventually resulting                              constraints through inpainting or RL control [Shi et al. 2024]. Simi-
in a latent motion that can be decoded [Meng et al. 2025; Pinyoa-                                larly, CAMDM [Chen et al. 2024] and PRIMAL [Zhang et al. 2025]
nuntapong et al. 2024a]. Some tokenized approaches also support                                  denoise a small window of future frames given a handful of past
precise kinematic controls through test-time-optimization [Pinyoa-                               frames. CAMDM is conditioned on a future trajectory to follow
nuntapong et al. 2025; Wan et al. 2024]. Besides masked models,                                  while PRIMAL relies on guidance and an additional ControlNet
several approaches take inspiration from language models [Radford                                for velocity, heading, and waypoint control. While CAMDM and
et al. 2018] and use autoregressive transformers to generate a se-                               PRIMAL show action label conditioning, none of these methods
quence of motion tokens that are decoded to human poses [Fan et al.                              support complex text prompting. UniPhys [Wu et al. 2025] enables
2025; Jiang et al. 2024a; Lu et al. 2025; Zhang et al. 2023]. While                              text control, but relies entirely on test-time guidance for kinematic
these methods are in fact autoregressive, they are generally large                               controls, which is inefficient for interactive applications. Closest
and slow models, designed for offline motion generation without                                  to our work is DiP [Tevet et al. 2025], which extends CAMDM by
support for precise kinematic control.                                                           adding conditioning on text and 3D target joint locations provided
   Our method ARDY delivers text-following and kinematic control                                 every two seconds. However, DiP’s short history and prediction
capabilities on par with recent offline models, while operating within                           horizon limit its ability to handle complex text prompts that require
an interactive framework. This is achieved through a novel two-                                  longer history context, and prevent it from satisfying kinematic
stage diffusion architecture that denoises a hybrid combination of                               constraints beyond its short generation horizon.
latent (tokenized) body and explicit root representations.                                          Latent diffusion has also been leveraged for interactive motion
                                                                                                 generation [Cen et al. 2025; Xiao et al. 2025; Zhao et al. 2025a].
   Interactive Motion Generation. Early works in autoregressive mo-                              DartControl [Zhao et al. 2025a] uses a VAE to learn a continuous
tion modeling leveraged non-linear latent variable models [Tay-                                  latent representation of motion primitives, then a diffusion model
lor et al. 2006] and recurrent neural networks [Fragkiadaki et al.                               that predicts future motion in this latent space. Similar to DiP, Dart-
2015]. Non-generative autoregressive prediction models [Holden                                   Control is limited by a short history context, and kinematic control
et al. 2017; Starke et al. 2022, 2019] have been trained for reac-                               such as 2D waypoint reaching or full-body in-betweening requires
tive character control by conditioning on various combinations                                   test-time-optimization or training an additional RL control policy.
of past and future poses and trajectory information. In parallel,                                MotionStreamer [Xiao et al. 2025] also learns a continuous latent
data-driven interactive animation systems such as Learned Motion                                 space using a causal convolutional autoencoder, then trains a causal

                                                                                                              ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:4   •    Zhao et al.


transformer denoiser to generate the next latent conditioned on
                                                                                                                                         Decoder
the past and text input. Similar to our approach, MotionStreamer is                                                                                    Body motion

trained on variable history length, making it more robust to complex                      Encoder                                                          ......
                                                                                                     Latent body motion
prompts. However, it lacks support for kinematic goal constraints.                                                                                      Un-patchify
                                                                                                           ......
   Several autoregressive diffusion models have been paired with
                                                                                                                                                    Causal Transformer
physics-based controllers to carry out generated motions in simula-
                                                                                                    Causal Transformer
tion [Huang et al. 2025; Rempe et al. 2023; Ren et al. 2023; Tevet et al.                                                                            Root Global2Local

2025; Wu et al. 2025]. Fully physics-based runtime character control                                      Patchify

is also an active area of study [Luo et al. 2023; Peng et al. 2022],                                       ......                                          ......            Hybrid
                                                                                                                                                                          representation
which has recently enabled both kinematic control and preliminary                                       Body motion
                                                                                                                                                          Patchify
text prompting [Tessler et al. 2024; Wu et al. 2025].                                                                                                      ......
                                                                                                                                                     Global root motion
   As shown in Tab. 1, our approach enables real-time generation
with native support for online text prompting, variable-length his-
tory contexts, and flexible long-horizon kinematic constraints—a
                                                                                    Fig. 2. Motion Tokenizer. The encoder first embeds the patchified body
combination of capabilities unmatched by prior works.                               motion into a latent representation. This latent body motion is concatenated
                                                                                    with the patchified global root motion to form our hybrid representation,
3      Method: ARDY                                                                 which is decoded back to reconstruct the body motion.
Our method ARDY consists of two main components: (1) a mo-
tion tokenizer first learns a compact latent representation of body
                                                                                    feature with a latent embedding. Concretely, a single pose x of a
motion, and then (2) an autoregressive two-stage motion diffusion
                                                                                    motion using the hybrid representation is a tuple
model learns to denoise hybrid motion tokens containing latent
body motion and explicit root motion. Our hybrid representation                                                            x = (mroot, xbody )                                             (2)
is introduced in Sec. 3.1 followed by the body motion tokenizer                     where xbody       ∈ R𝐿 is the latent body representation with dimension-
in Sec. 3.2. The autoregressive generation problem formulation is                   ality 𝐿, which has replaced mbody from the explicit representation.
detailed in Sec. 3.3 and then the diffusion model that solves it is                 In practice, xbody is the output of a learned tokenizer (Sec. 3.2) and
described in Sec. 3.4. Finally, Sec. 3.5 covers implementation details.             each token encodes multiple frames of motion. The diffusion model
                                                                                    introduced in Sec. 3.4 learns to generate motion using the hybrid
3.1        Hybrid Motion Representation                                             representation, which has several advantages. Maintaining root po-
To balance the representational compactness required for efficient                  sition features in global coordinates avoids potential compounding
generative learning with the need for direct, precise control via                   errors inherent to integrating local velocity-based representations.
explicit feature overwriting, we propose a hybrid motion represen-                  The global root also facilitates controllable motion generation con-
tation that decouples root motion from body motion. Specifically,                   ditioned on spatial constraints, which are often sparse and defined
root trajectories are represented in an explicit, interpretable form,               within the global scene space, as it enables direct overwriting of root
while body motion is encoded in a compact latent space. In this                     features. Moreover, the latent body representation is more compact
section, we give a high-level overview of the hybrid motion repre-                  than explicit representations, and pre-defined after the tokenizer is
sentation and its advantages for generation before detailing how                    trained. This makes it better suited for generative modeling, both
the latent component is learned in Sec. 3.2.                                        computationally and in terms of learning efficiency.
  Explicit Motion Representation. Our hybrid representation builds                  3.2     Body Motion Tokenizer
on an explicit motion representation, which we describe first for
context. Each frame of a motion that uses this explicit representation              We train a motion tokenization network to compress the high-
m = (mroot, mbody ) ∈ R𝑀 is defined as a tuple of root and body                     dimensional explicit body features into a compact latent space, facil-
skeleton joint features                                                             itating more efficient generative learning. As illustrated in Fig. 2, the
                                                                                    tokenizer employs an asymmetric conditional autoencoder archi-
            mroot = (p, cos𝜓, sin𝜓 ) ∈ R5,          mbody = (𝜽, J, ¤J, c),    (1)   tecture. Given an explicit body motion m1:𝑁 body
                                                                                                                                     containing 𝑁 frames,
                                                                                    we treat each 𝑃 consecutive frames as a patch by reshaping them
where p ∈ R3 denotes the global root position, 𝜓 ∈ (−𝜋, 𝜋] denotes
                                                                                    into a single vector, resulting in 𝑇 = 𝑁 /𝑃 input vectors to the en-
the root heading angle, 𝜽 ∈ R6𝑗 denotes the 6D representation
                                                                                    coder. The encoder compresses the body motion into latent tokens
[Zhou et al. 2019] of the global joint rotations for all 𝑗 skeleton
                                                                                    x1:𝑇   ∈ R𝑇 ×𝐿 , which are then concatenated along the feature dimen-
joints including the root, J ∈ R3𝑗 −3 denotes the non-root joint                      body
                                                                                                                                              𝑇 ×5𝑃 to form
positions subtracted by the planar root position, J¤ ∈ R3𝑗 denotes the              sion with the patchified explicit root motion m1:𝑇root ∈ R
global joint velocities, and c ∈ R4 denotes the binary floor contact                the hybrid motion tokens:
label for the feet joints. The explicit representation feature size 𝑀                                                     x1:𝑇 = [m1:𝑇     1:𝑇
                                                                                                                                   root ; xbody ]                                          (3)
depends on the number of joints in the skeleton.
                                                                                    resulting in x1:𝑇          ∈ R𝑇 ×𝐷 where 𝐷 = 𝐿 + 5𝑃. The decoder subse-
   Hybrid Motion Representation. Our hybrid motion representation                   quently reconstructs the body motion from these hybrid tokens.
is formed by simply replacing the body component of the pose                        Crucially, the decoder first transforms the global root motion from

ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                             ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation       •   86:5


Eq. (1) into a local representation, which replaces the global root                        While autoregressive generation maintains temporal causality
motion for the conditional input to the decoder network. Each root                     (i.e., there is no dependence on future frames), it can still be condi-
pose in the local representation is a tuple (𝜓,¤ ¤p𝑥 , ¤p𝑧 , p𝑦 ) where 𝜓¤ is          tioned on future goals g1:(𝐶+𝐹 ) . Spatial goals encompass constraints
the 1D angular velocity of the heading, ¤p𝑥 and ¤p𝑧 are the x and z                    on the motion that specify joint position and/or rotation values at
components of the linear root velocity, and p𝑦 is the 𝑦-component                      specific timesteps in the future. These can be used to hit 2D way-
(height) of the root. Note that while the global root representation                   points or follow full paths on the ground, full-body pose keyframes,
is useful for generating motion as discussed previously, in the tok-                   sparse end-effector position constraints, and more. Importantly,
enizer decoder we find the local representation is more suitable to                    our formulation is not limited to goals within the current predic-
significantly mitigate foot skating (discussed in Sec. 5.2 and Tab. 2).                tion window, but is also conditioned on goals further in the future.
   We use transformer encoder layers with causal attention in both                     Out-of-window goal constraints implicitly determine the motion
the encoder and decoder, which ensures that each frame embedding                       generation within the current window, even though they do not
relies only on preceding frames and preserves temporal causality. We                   directly apply to the immediate frames. For example, when a hu-
experimented with different autoencoder variants for the tokenizer,                    man needs to run to a location in 10 seconds, the destination goal
including variational autoencoder (VAE) [Kingma and Welling 2014]                      will determine in which direction the human should start moving
and finite scalar quantization (FSQ) [Mentzer et al. 2023] variants,                   from the first step. Supporting such long-horizon goals in previous
as detailed in Sec. 5.3. While all variants perform similarly, we find                 works requires training an additional RL control policy on top of
that FSQ demonstrates better stability in training, making it the                      the autoregressive motion model [Shi et al. 2024; Zhao et al. 2025a],
default tokenizer choice. Training details can be found in Sec. 3.5.                   while our model architecture supports this natively.

3.3   Controllable Interactive Motion Generation                                       3.4    Autoregressive Two-Stage Diffusion Model
We aim to develop a motion generation model that supports text                         Based on the hybrid motion representation, we design a transformer-
and spatial conditions from real-time input streams. At runtime,                       based diffusion denoising model to learn the goal-conditioned au-
the model should be reactive to any changes in the input streams                       toregressive motion generation task. To further enable precise con-
like a new text prompt or shift in goal location. Similar to prior                     trollability without sacrificing motion fidelity, we introduce an in-
work [Chen et al. 2024; Tevet et al. 2025; Wu et al. 2025; Zhang et al.                terleaved two-stage diffusion framework that decomposes the gen-
2025], we formulate this problem as a conditional autoregressive                       eration of root motion and body motion.
generation task that synthesizes a short window of future motion                          For an introduction to human motion diffusion, we refer the
starting from the current frame, conditioned on past history and                       interested reader to prior work [Tevet et al. 2023; Zhang et al. 2024a],
optional goal inputs (i.e., kinematic constraints). The synthesized                    and focus here on relevant details for our method. At step 𝑘 in the
future motion is then played back for the user until re-planning                       denoising process, our diffusion model takes 𝐶 noisy hybrid motion
occurs and the model predicts future motion in the new window.                         tokens x𝑘1:𝐶 ∈ R𝐶 ×𝐷 within the current generation window, along
   Our autoregressive model operates in the hybrid token space.                        with relevant conditioning, and outputs a prediction for the clean
Assuming that the prediction window starts at token index 1, then                      denoised hybrid tokens x̂1:𝐶
                                                                                                                 0 . Mirroring Eq. (4), the denoising process
our goal is to train the generative model F to generate the next 𝐶                     at step 𝑘 can be written as:
tokens in the current prediction window:
                                                                                                                        1:𝐶 (−𝐻 +1):0 1:(𝐶+𝐹 )
                                                                                                        x̂1:𝐶
                                                                                                          0 = F (𝑘, 𝑠, x𝑘 , x        ,g        ).                        (5)
                    x1:𝐶 = F (𝑠, x (−𝐻 +1):0, g1:(𝐶+𝐹 ) ),                  (4)
                                                                                       The high-level architecture of the denoising network is illustrated
                                                                                       in the left side of Fig. 3. The diffusion step and text conditioning
where 𝑠 is the text prompt describing the motion semantics of the                      are each a single token fed in alongside the sequence of history
current generation window, x (−𝐻 +1):0 is the history motion span-                     tokens and noisy tokens for the current prediction window. We use
ning up to the previous 𝐻 tokens, and g1:(𝐶+𝐹 ) denotes the spatial                    sinusoidal positional encodings for motion tokens to embed their
goals to achieve. Note that the goals for the first 𝐶 tokens g1:𝐶 are                  temporal position within the motion sequence, while employing
within the current prediction horizon, while g (𝐶+1):(𝐶+𝐹 ) are goals                  separate learned positional embeddings for text and diffusion tokens.
beyond the prediction window, up to 𝐹 additional future tokens.                        Linear layers are used to project all token types to the same feature
   Notably, 𝐻 can vary in our formulation, so the model should                         dimensionality before feeding to the denoiser.
expect to receive anywhere from 0 to a maximum of 𝐻 history con-
ditioning tokens. A long history context is crucial to handle text                       Spatial Goal Conditioning. We represent spatial goal inputs g with
prompts that describe complex non-cyclic motions. For instance, the                    a masked version of the explicit motion representation from Eq. (1).
prompt “walk forward, then bend over and pick something up before                      This allows handling arbitrarily sparse global signals on any pose
continuing to walk” has walking before and after the pick-up action.                   feature, such as keyframed body or end-effector joints. Only the
In autoregressive formulations with limited history context [Tevet                     constrained features and timesteps in g contain non-zero values
et al. 2025; Wu et al. 2025; Zhao et al. 2025a], a model conditioned                   while other unconstrained entries are set to zero. We additionally
only on recent walking frames cannot determine whether a preced-                       define a corresponding binary mask v of the same shape, which
ing pick-up action has already occurred or still needs to be generated,                indicates the dimensions that are constrained. To align with the
leading to inaccurate generations with missing or duplicated actions.                  temporal granularity of the motion tokens, we assume the goal

                                                                                                  ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:6   •   Zhao et al.


                                        Clean hybrid motion prediction                                                         Two-stage Denoiser
                                                      ......
                                                                                                              Clean                                                         Clean latent
                                                                                                            global root                                                     body tokens
                                                                                                               ......                                                          ......
                      Two-Stage Transformer Denoiser

               -H+1                          0    1            C   C+1                 C+F
                                                                                                      Root Transformer                                           Body Transformer
                           ......                     ......                  ......

                                                                                                       1                   C                                       1                       C
   Step Text          Variable length            Overwrite root     Variable length future
                                                                                                            Noisy latent                                                    Noisy latent
                      hybrid history               & concat          spatial constraints
                                                                                             ......         body tokens         ......                  ......              body tokens        ......
                                                                                                               ......                                                          ......
                                        ......                       ......
                                                                                                              Noisy                                                           Clean
                                                                                                            global root                                                     global root
                         Noised current hybrid tokens Current spatial constraints



Fig. 3. Autoregressive Two-Stage Transformer Denoiser. (Left) Conditioned on a variable-length history context and optional spatial goal constraints, the
autoregressive denoiser predicts a sequence of 𝐶 clean motion tokens within the current generation window. Spatial goal constraints can be arbitrarily sparse
and may be located within or beyond the current motion generation window. (Right) The two-stage denoiser first predicts clean global root motion, which
then conditions the second stage to predict clean latent body tokens, together forming the complete hybrid motion prediction.


inputs are patchified, for example the short-term goals are g1:𝐶 ∈                                    predicted hybrid motion representation is processed by the tok-
R𝐶 ×𝑀𝑃 with patch size 𝑃 and pose feature dimensionality 𝑀.                                           enizer’s decoder to recover the explicit body motion and form the
   Before being given to the model, the root part of the noisy tokens                                 full, un-patchified explicit motion as m̂1:𝐺      1:𝐺     1:𝐺
                                                                                                                                               0 = [ m̂root ; m̂body ], where
m1:𝐶
  root is overwritten with the root component of the constraint as                                    the generation window size in frames is 𝐺 = 𝐶 · 𝑃.
m̃1:𝐶                     1:𝐶               1:𝐶
  root = (1 − vroot ) ⊙ mroot + vroot ⊙ groot where ⊙ is the element-                                    Our two-stage architecture is motivated by the hypothesis that
wise product. This root constraint overwriting [Rempe et al. 2026;                                    predicting body motion conditioned on clean root motion is an easier
Setareh et al. 2024] facilitates highly accurate control over the root                                task than generating both root and body jointly. This decomposition
trajectory, which governs the fundamental global movement of the                                      is designed to enable precise controllability without compromising
human motion. To incorporate constraints on detailed body poses                                       the fidelity of the synthesized motion. As demonstrated in our ab-
and make the model aware of all constraints, we concatenate the                                       lation study in Tab. 2, the proposed two-stage architecture yields
explicit body goal features and the full constraint mask with the                                     better results compared to a monolithic one-stage baseline that
input tokens along the feature dimension. In other words, the in-                                     simultaneously predicts root and body motion.
put noisy tokens are extended with masked constraints to form
the augmented representation [ m̃1:𝐶         1:𝐶    1:𝐶              1:𝐶
                                    root ; xbody ; gbody ; v] where xbody is                          3.5     Training and Implementation Details
the latent body part of the input noisy tokens. Since there are no                                       Motion Tokenizer. In practice, our motion tokenizer uses a patch
noisy input tokens beyond the prediction horizon 𝐶, the patchified                                    size of 𝑃 = 4 frames. Both the encoder and decoder are implemented
long-horizon constraints g (𝐶+1):(𝐶+𝐹 ) ∈ R𝐹 ×𝑀𝑃 are simply concate-                                  as 8-layer transformers with a latent dimension of 512, utilizing
nated with their corresponding binary mask and fed in as additional                                   causal self-attention to preserve temporal consistency. The tokenizer
tokens to the transformer. These long-horizon goal tokens can vary                                    is trained on motion clips of varying lengths (1–10 seconds) using a
in length and sparsity depending on user input, with unconstrained                                    reconstruction loss and additional loss penalizing foot skating:
tokens masked out during transformer inference.
                                                                                                                                                    Í                  ¤̂
                                                                                                                                                        𝑗 ∈ S𝑓 ĉ 𝑗 ∥ J 𝑗 ∥ 2
   Interleaved Two-Stage Denoiser. Our autoregressive transformer                                                                        Lskate =        Í                       ,                      (6)
                                                                                                                                                             𝑗 ∈ S𝑓 ĉ 𝑗
denoiser employs an interleaved, two-stage design [Rempe et al.
2026] to sequentially predict clean root and body motions. The inter-                                 where S𝑓 represents the set of foot joint indices, ĉ 𝑗 denotes the
nals of our transformer-based denoiser are shown on the right side                                    predicted contact label for foot joint 𝑗, and ∥ J¤̂ 𝑗 ∥ 2 denotes the magni-
of Fig. 3. At each denoising step, the model first predicts the explicit                              tude of predicted foot joint velocity. This foot-skating loss penalizes
clean global root motion m̂1:𝐶   root with the root transformer. Next, the                            the velocities of joints predicted to be in contact with the ground,
global root motion is detached and fed into the body transformer,                                     thereby enforcing stationary constraints during the contact phase.
which predicts the clean latent body tokens x̂1:𝐶   body
                                                         . The outputs from                           We set the weight for this loss term to 0.01. The exact implemen-
both branches are concatenated to form the clean hybrid motion                                        tation of the reconstruction loss depends on the framework being
prediction x̂1:𝐶      1:𝐶      1:𝐶
              0 = [ m̂root ; x̂body ]. During inference, this concatenated                            employed for the tokenizer. We test three different approaches in-
hybrid prediction is re-noised for the subsequent diffusion step and                                  cluding a vanilla continuous autoencoder, VAE, and finite scalar
fed back into the two-stage denoiser. This iterative and interleaved                                  quantization (FSQ) [Mentzer et al. 2023] and compare them in exper-
denoising process ensures continuous mutual influence between                                         iments later (Sec. 5.3). For the FSQ variant, we apply finite quantiza-
the root and body transformers throughout generation. Finally, the                                    tion to the encoder output embedding, constraining each feature to

ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                        ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation       •   86:7


one of 64 discrete levels. These quantized vectors serve directly as
the latent representation. For all tokenizer variations, we train with
the AdamAtan2 [Everett et al. 2024] optimizer for 4 million steps
using a learning rate of 2𝑒−5 and batch size of 128. We employ a
cosine learning rate scheduler with a 10k-step linear warmup phase.
Training is performed on a single NVIDIA A100-SXM4-80GB GPU.
   Two-Stage Denoiser. Both the root and body transformer in our
two-stage denoiser employ the same transformer encoder architec-
ture. Each transformer contains 8 layers with 8 heads and a latent
size of 1024, totaling around 156 million parameters for our deployed
denoiser model in the interactive demo. For text encoding, we use
LLM2Vec [BehnamGhader et al. 2024], which is an embedding model
trained on top of Llama-3-8B-Instruct [AI@Meta 2024].
   After training the tokenizer, we train the denoiser using the
DDPM framework [Ho et al. 2020] with a modified version of the                    Fig. 4. Interactive Demo Interface. This web interface allows generating
“simplified” loss function that contains several components. In the               motion and interacting with ARDY in real time. The control panel at the
following discussion, we drop the token/frame index superscripts                  top right allows dynamically changing the text prompt or input constraints.
from all terms for simplicity. First, given the clean hybrid prediction           Input constraints are visualized in red within the 3D scene as the model
x̂0 = [m̂root ; x̂body ] and ground truth x0 , the hybrid loss                    generates motion to follow them. The timeline tracks on the bottom of the
                                                                                  interface intuitively show upcoming text prompts and constraints.
                         Lhybrid = || x̂0 − x0 || 1                    (7)
uses a smooth L1 loss [Girshick 2015] to penalize errors between
the predicted and ground truth hybrid motion tokens. For the next                 sparse keyframes, full-body keyframe blocks, sparse end-effector
loss, we decode the predicted tokens with the tokenizer decoder                   keyframes, and foot contact keyframes. To enable classifier-free
D resulting in the predicted explicit body motion m̂body = D ( x̂0 ).             guidance [Ho and Salimans 2021] during inference, we randomly
Then, the decoded body loss                                                       drop the text prompts and spatial constraints with a 10% probability.
                      Ldec = || m̂body − mbody || 1                    (8)           By default, we use ten diffusion steps during both train and test-
                                                                                  time, which strikes a good balance between speed and accuracy.
compares the predicted explicit body motion to the ground truth                   However, performance is still acceptable for most applications when
mbody . To place greater emphasis on accurately hitting the specified             going as low as four steps (see Sec. 5.3). Denoiser training uses the
constraints, we add a goal loss                                                   AdamAtan2 optimizer with a learning rate of 2𝑒−5. Importantly, we
                       Lgoal = ||v ⊙ ( m̂0 − g)|| 1                    (9)        do not use dropout in the denoiser as this causes root constraint
                                                                                  conditioning inputs to be partially lost. Our denoiser models are
that specifically penalizes components in the full explicit motion
                                                                                  trained with a batch size of 512 across four NVIDIA A100-SXM4-
prediction m̂0 that do not hit the constraint goals in g. Finally, we add
                                                                                  80GB GPUs for one million optimization steps.
a regularizer to ensure consistency between the directly predicted
joint positions and those resulting from the predicted joint rotations
                                                                                  4     Interactive Motion Generation Demo
via forward kinmeatics:
                                                                                  To showcase ARDY’s versatility, we developed an interface using
                      Lconsist = || Ĵ0 − FK(𝜽ˆ 0 )|| 2               (10)        Viser [Yi et al. 2025] to interactively generate motion with our model.
where Ĵ0 denotes the predicted joint positions, and the forward                  The system, shown in Fig. 4, enables real-time character control
kinematics function (FK) outputs joint positions given the predicted              through a combination of streaming text prompts and interactive
joint rotations 𝜽ˆ 0 . The final loss combines all these objectives as            spatial constraints provided via mouse and keyboard inputs. In this
                                                                                  section, we first detail ARDY’s test-time operation, then qualitatively
                L = Lhybrid + Ldec + Lgoal + Lconsist .               (11)        demonstrate key results through the interactive demo.
   The two-stage denoiser is trained on sequences with a maximum
length of 10 seconds following existing offline motion generation                 4.1    Test-Time Operation
works [Pinyoanuntapong et al. 2025; Tevet et al. 2023]. For each train-           During inference, ARDY operates autoregressively to synthesize
ing motion sequence, a fixed-size generation window of 𝐺 frames                   motion in response to a dynamic stream of user inputs. In the first
is sampled randomly. Consequently, the lengths of the available                   step of the motion roll-out, ARDY generates the first window of
history (𝐻 ) and future (𝐹 ) contexts for each training sample vary dy-           length 𝐺 with no history poses as input. In subsequent steps, the
namically, ranging from 0 to the maximum sequence length minus 𝐺.                 previously predicted tokens become the history conditioning as
Moreover, we augment the motion sequences by applying random                      the model predicts the next window of 𝐺 motion frames. To facili-
rotations around the 𝑦-axis. Spatial constraints for both in-horizon              tate autoregressive long motion generation, we employ a truncated
and out-of-horizon are randomly sampled from a set of common use                  sliding window to manage both historical and beyond-generation
cases including 2D root keyframes, 2D root trajectories, full-body                future contexts. The specific truncation lengths of these context

                                                                                             ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:8   •    Zhao et al.


              Replan trigger
                                                                                                             that the system can process complex descriptions and seamlessly
0                    i         i+B    N                                                                      adapt to dynamic changes in user-specified text prompts. It also
                                           Concat after buffer
            Motion sequence                                                                                  robustly satisfies diverse kinematic constraints, ranging from sparse
                    Replan buffer                        0                        i    i+B        N'=i+B+G
                                                                                                             long-term goals (e.g., reaching a target location in 10 seconds) to
                                                                 Motion until buffer    New generation
                                                                                                             dense short-term constraints (e.g., trajectory following or full-body
       History condition                                                                                     keyframes). Additional qualitative results for kinematic constraint-
                               i+B          i+B+G                                                            conditioned generation are shown in Fig. 6.
            Model               New generation                                                                  Our system also supports diverse locomotion interfaces: users can
                                                                                                             define target root trajectories in real time using mouse-based way-
Fig. 5. Latency-Aware Replanning. We utilize a non-blocking strategy                                         points or modulate real-time velocity via keyboard commands. For
where a buffer of 𝐵 frames is simultaneously played back and fed into                                        mouse-based root path control, we derive the target trajectory by
the generation thread as history context. This buffer effectively hides the                                  linearly interpolating between mouse-click waypoints and smooth-
inference latency of slower models, ensuring that the transition to the newly                                ing the resulting path. For keyboard-based root velocity control, we
generated sequence remains smooth and continuous.                                                            compute a target velocity from user input and the current velocity,
                                                                                                             then linearly interpolate between the two and integrate the resulting
                                                                                                             per-frame velocities to derive the root trajectory input to the model.
windows are configurable in our interactive demo, up to a maximum                                            Extensive video demonstrations of our interactive generation sys-
of 8 seconds—a limit established by the longest context observed                                             tem are provided in the supplementary material, highlighting its
during training. Future constraints that fall beyond the truncation                                          responsiveness and high-fidelity motion quality.
limit (e.g., a target location one minute ahead) are excluded from the
input constraint tokens. They are only incorporated into the condi-                                          5     Analysis on Large-Scale Mocap Data
tioning once the advancing generation window brings them within                                              Next, we thoroughly analyze key design choices of ARDY along
the truncated future context horizon. During the autoregressive                                              with the effects of various hyperparameter settings.
generation, the root component of the previously predicted tokens
are translated such that the last frame of the history coincides with                                        5.1    Experiment Setting
the origin, which is what the model expects as input. The transla-
tion offset is preserved and subsequently applied to the generated                                              Bones Rigplay Mocap Dataset. We leverage the large-scale propri-
motion to transform it back into global scene coordinates. This loop                                         etary Bones Rigplay dataset [Bones Studio 2026], which contains
ensures high-quality motion with smooth temporal transitions.                                                around 700 hours of diverse studio-quality human motion with text
   To enable real-time interactivity, we incorporate a dynamic re-                                           descriptions. The scale and quality of this data enables a more ro-
planning mechanism that triggers immediately upon detecting new                                              bust testbed for evaluating design variations compared to smaller
user input, such as updated text prompts or modified future kine-                                            public datasets like HumanML3D [Guo et al. 2022], which are sat-
matic constraints, or when the current motion buffer will soon be                                            urated as indicated by methods scoring higher than ground truth
depleted. Our replanning scheme is latency-aware, facilitating the                                           data on metrics like R-precision. This dataset contains motions from
use of more powerful models even when their inference latency                                                more than 150 participants and is retargeted to a unified-proportion
exceeds the inter-frame interval. As shown in Fig. 5, when a re-                                             27-joint skeleton to facilitate learning. The motions encapsulate
plan is triggered we utilize the subsequent 𝐵 frames, which have                                             thousands of distinct behaviors, each performed by multiple actors
already been generated, as a replan buffer. These frames are played                                          for multiple takes, resulting in a diverse distribution of semantics
back to the user while simultaneously serving as history context for                                         and kinematic variations. It includes common motion categories
the asynchronous generation thread. This replan buffer effectively                                           such as locomotion, everyday activities, gestures, and combat, per-
masks the inference latency of slower models, ensuring smooth                                                formed in a variety of styles. Raw motion clips range from 1 to 180
and continuous transitions to the new generation. We present this                                            seconds in length, but we clip motions to a maximum of 10 seconds
scheme as an optional mechanism to enable increased diffusion                                                and subsample to 20 fps for training. To improve generalization,
steps for enhanced motion quality and control accuracy. In our de-                                           we use LLM to generate diverse paraphrases of the original text
ployment setup, the 4-step model operates without buffer frames,                                             labels. The dataset is split into training and test sets by first group-
while the 10-step model employs a single buffer frame.                                                       ing motion clips according to semantic content (i.e., action type,
                                                                                                             such as “eating_apple_right”), and then assigning disjoint groups to
4.2        Demo Results                                                                                      each split with an approximate 90/10 ratio, resulting in about 315k
                                                                                                             motion clips for training and 35k for testing. As a result, the test set
The interactive motion generation demo uses ARDY trained on the                                              contains motion categories that are entirely unseen during training,
Bones Rigplay dataset [Bones Studio 2026] described in detail later                                          providing a stronger evaluation of generalization to novel actions.
(see Sec. 5). The demo runs on a workstation equipped with an RTX
4090 GPU. The average generation latency is 33 ms for our efficient 4-                                         Constraints Sampling. We evaluate text+constraint-conditioned
step diffusion model and 63 ms for our 10-step diffusion model, with                                         generation across a comprehensive suite of test cases designed to
the latter providing slightly improved control accuracy. Both models                                         simulate common downstream applications. These scenarios include
use a generation window of 𝐺 = 40 frames (2 seconds at 20 fps). All                                          dense root trajectory following, sparse waypoints navigation, full-
examples in Fig. 1 are generated using this interface, demonstrating                                         body keyframes, and end-effector joints control (incorporating both

ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                                       ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation                              •     86:9



                   Root trajectory                                           Root waypoints                                      Full body Keyframes                  Joints position and rotation




     "A person walks forward at a steady pace, then
                                                         "Individual energetically steps right and left while performing   "A ballet dancer, performs a forward       "A person picks up an object from
     turns to face the left diagonal, jogs sideways to
     the right, and waves their right hand."             high jump and striking arm motion on the fourth step."            turn joining feet, in a repeating loop."   the front at a high level in front,
                                                                                                                                                                      places it at a low level in front."




   "A person walks and turns unsteadily and painfully     "A person is running, jumping onto a bike, and riding it."       "A fatal back hit with a sword,                 "Performing a front knee
   in multiple directions with an injured torso."                                                                          assumedly leading to falling in agony."         kick to the left."


Fig. 6. Motion Generation with Kinematic Constraints. Qualitative results for motion generation conditioned on text prompts and diverse kinematic
constraints, including dense root trajectories, sparse root waypoints (visualized as red rings), full-body keyframes (visualized as red skeletons), sparse joint
positions (visualized as white skeletons with constrained joints highlighted as red spheres), and joint rotations (visualized as coordinate axes centered at the
constrained joint). Motion temporal progression is indicated by a color gradient from gray to blue.


position and orientation goals). The spatial constraints are sampled                                    5.2      Ablation Study
directly from the ground-truth test set alongside their correspond-                                     Tab. 2 presents ablation results on three key design choices: the
ing text prompts. Furthermore, to rigorously evaluate the model’s                                       hybrid motion representation, the global-to-local root conversion
robustness against constraint inputs, we introduce slight random                                        within the tokenizer decoder, and the two-stage denoiser design.
perturbations to the global translation and heading of a subset of
sampled constraints during the evaluation.                                                                 Hybrid Motion Representation. We first compare our proposed
                                                                                                        hybrid motion representation (derived via the learned tokenizer)
                                                                                                        against the purely explicit motion representation. To ensure a fair
   Evaluation Metrics. Following established protocols [Guo et al.                                      comparison, we train an autoregressive baseline that uses explicit
2022], we employ Fréchet Inception Distance (FID) to quantify the                                       pose features, applying the same patching strategy to align the tem-
distributional similarity between generated and ground-truth mo-                                        poral granularity of the tokens. This explicit baseline uses masked
tions, and Top-3 R-precision to assess text-motion alignment. To                                        overwriting (Sec. 3.4) to condition on all kinematic constraint inputs
ensure a rigorous evaluation, we train a robust evaluator model                                         by overwriting both constrained root and body features. As demon-
based on TMR [Petrovich et al. 2023] using the large-scale Bones                                        strated in Tab. 2, our autoregressive model utilizing the hybrid
Rigplay dataset. Notably, we compute R-precision over a test dataset                                    representation significantly outperforms its explicit counterpart in
containing about 5k unique samples of diverse action types. This                                        both motion quality and control accuracy. The high-dimensionality
significantly increases the retrieval difficulty compared to the stan-                                  of explicit motion representations likely complicates the genera-
dard practice in benchmarks like HumanML3D [Guo et al. 2022],                                           tive learning process, particularly under our few-step denoising
which computes the metric over batches of size 32 only. As a proxy                                      setting. In contrast, the hybrid representation compresses high-
for motion quality, we also report a heuristic foot skating met-                                        dimensional body features into compact latent embeddings that are
ric that measures mean foot velocity when the foot is considered                                        more amenable to efficient generative modeling.
in-contact based on a height threshold. To assess spatial control
accuracy in constraints-conditioned generation, we compute the                                            Global-to-Local Conversion. Next, we evaluate the importance of
mean error between the user-specified constraint targets (position                                      our global-to-local root conversion process within the tokenizer
and orientation) and the corresponding generated poses.                                                 decoder by training a baseline decoder that operates directly on the

                                                                                                                       ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:10   •   Zhao et al.


Table 2. Quantitative Ablation of Architectural Designs. We evaluate performance across text-only and various kinematic constraints-conditioned
generation scenarios, including end-effector joint rotation and position, full-body keyframe joints, dense root trajectories, and sparse root waypoints. ↑
denotes higher values are better; ↓ denotes lower values are better. Bold and underlined values indicate the best and second-best results, respectively.

                                              Text-only Generation                                                          Constraints-conditioned Generation
  Model                                 Skate (m/s) ↓ R-prec. ↑ FID ↓             Skate (m/s) ↓     Joint rot. (deg.) ↓     Joint pos. (m) ↓ Keyframe body (m)↓         Traj. (m) ↓      Waypoint (m) ↓
  Dataset                                     0.255         76.56      0.000            -                       -                    -                     -                 -                 -
  ARDY (Ours)                               0.264           65.47      0.027         0.250                  2.23                   0.025                0.023              0.015             0.024
  Explicit representation                   0.365           53.90      0.065         0.281                  1.67                   0.130                0.136              0.033             0.203
  Global root-conditioned decoder           0.303           64.94      0.028         0.284                  2.88                   0.048                0.044              0.024             0.060
  One-stage architecture                    0.264           65.84      0.029         0.248                  2.46                   0.101                0.079              0.017             0.164


Table 3. Hyperparameter and Tokenizer Analysis. The best results in each group are highlighted in bold, and the second best are underlined. The ablation
table is divided into five sections, sequentially comparing: (1) generation horizon, (2) diffusion steps, (3) tokenizer patch sizes, (4) tokenizer latent space
capacities (latent embedding quantization levels and dimensions), and (5) various tokenizer types. The default configuration in each section is marked with ∗ .

                        Model                              Text-only Generation                                                    Constraints-conditioned Generation
 Tokenizer                Horizon   Diffusion step    Skate (m/s) ↓   R-prec. ↑    FID ↓    Skate (m/s) ↓    Joint rot. (deg.) ↓   Joint pos. (m) ↓   Keyframe body (m)↓   Traj. (m) ↓    Waypoint (m) ↓
                                                                                            Generation horizon
 FSQ 64-128, Patch 4         4           10              0.151         33.42      0.224         0.445               9.23                 0.848                 0.864         0.864            0.850
 FSQ 64-128, Patch 4         8           10              0.258         56.70      0.037         0.243               3.45                 0.031                 0.026         0.013            0.020
 FSQ 64-128, Patch 4        12           10              0.254         59.54      0.033         0.247               2.94                 0.033                 0.028         0.017            0.031
 FSQ 64-128, Patch 4        20           10              0.255         63.80      0.030         0.250               2.61                 0.046                 0.037         0.014            0.059
 FSQ 64-128, Patch 4        40∗          10              0.264         65.47      0.027         0.250               2.23                 0.025                 0.023         0.015            0.024
                                                                                              Diffusion steps
 FSQ 64-128, Patch 4        40            1              0.411         56.74      0.079         1.405               25.39                1.040                 1.054         1.037            1.002
 FSQ 64-128, Patch 4        40            2              0.239         61.28      0.052         0.360               7.96                 0.174                 0.169         0.274            0.163
 FSQ 64-128, Patch 4        40            3              0.231         63.59      0.041         0.254               3.58                 0.053                 0.051         0.046            0.044
 FSQ 64-128, Patch 4        40            4              0.230         64.41      0.034         0.249               2.68                 0.034                 0.032         0.028            0.027
 FSQ 64-128, Patch 4        40           10∗             0.264         65.47      0.027         0.250               2.23                 0.025                 0.023         0.015            0.024
 FSQ 64-128, Patch 4        40           100             0.282         65.49      0.025         0.257               2.71                 0.030                 0.027         0.009            0.028
                                                                                            Tokenizer patch size
 FSQ 64-128, Patch 1        40           10              0.298         44.45      0.152         0.355               2.31                 0.764                 0.816         0.790            0.775
 FSQ 64-128, Patch 4∗       40           10              0.264         65.47      0.027         0.250               2.23                 0.025                 0.023         0.015            0.024
 FSQ 64-128, Patch 8        40           10              0.317         68.01      0.022         0.295               3.05                 0.070                 0.062         0.018            0.100
                                                                                     Tokenizer latent space capacity
 FSQ 16-32, Patch 4         40           10              0.283         68.11      0.023         0.261               4.57                 0.031                 0.026         0.016            0.020
 FSQ 64-32, Patch 4         40           10              0.273         67.62      0.025         0.252               3.96                 0.026                 0.023         0.014            0.017
 FSQ 64-128, Patch 4∗       40           10              0.264         65.47      0.027         0.250               2.23                 0.025                 0.023         0.015            0.024
 FSQ 64-256, Patch 4        40           10              0.268         64.04      0.031         0.257               2.31                 0.030                 0.025         0.015            0.032
                                                                                              Tokenizer type
 AE 128D, Patch 4           20           10              0.266         62.20      0.033         0.246               2.23                 0.044                 0.040         0.016            0.057
 VAE 128D, Patch 4          20           10              0.259         63.35      0.031         0.250               2.17                 0.046                 0.042         0.014            0.058
 FSQ 64-128, Patch 4∗       20           10              0.255         63.80      0.030         0.250               2.61                 0.046                 0.037         0.014            0.059



global root representation. The ablation results reveal that removing                                       5.3     Hyperparameter and Tokenizer Type Analysis
the global-to-local root conversion leads to a notable increase in                                          Tab. 3 provides an analysis of the generation horizon length, the
foot skating, confirming that local root representations are essential                                      number of diffusion steps, and the tokenizer configurations.
for preserving motion quality and physical plausibility.
                                                                                                               Generation Horizon. The generation horizon length is a critical
                                                                                                            hyperparameter impacting the model’s performance. We observe
                                                                                                            that extending the horizon consistently improves motion fidelity
                                                                                                            (FID) and semantic alignment (R-Precision) metrics. Conversely, ex-
   Two-Stage Denoiser Design. To validate our two-stage model ar-                                           tremely narrow horizons (e.g., 4 frames) lead to training instability
chitecture, we train a one-stage baseline that jointly predicts the                                         and degraded performance, ultimately resulting in the generation of
root trajectory and latent body motion tokens simultaneously. Ex-                                           drifting motions. The text-only foot-skating metric for the 4-frame
perimental results show that the two-stage architecture achieves                                            horizon is misleadingly low, as the model often fails to respond to
superior performance, yielding higher-fidelity text-conditioned mo-                                         text prompts. Regarding spatial control, we find that horizons of 8
tion and significantly lower spatial constraint errors. This suggests                                       and 40 frames effectively minimize the constraint errors. Qualitative
that decomposing root and body prediction facilitates the simultane-                                        analysis shows that models with an 8-frame horizon can transition
ous learning of high-fidelity generation and precise spatial control.                                       between actions more rapidly in response to updated text prompts

ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                    ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation                               •       86:11


compared to those with a 40-frame horizon. Furthermore, our ex-                Table 4. Offline Text and Constraint Control Comparison. Evaluation
periments show that the 8-frame model learns constraint adherence              results of text-conditioned motion generation with joint position goals on
                                                                               HumanML3D. ∗ denotes methods without test-time optimization. ↑ denotes
faster during training than its 40-frame counterpart.
                                                                               higher values are better; ↓ denotes lower values are better.
   Diffusion Step. We ablate the impact of the number of diffusion
                                                                                Method                                        R-Prec. ↑   FID ↓     Skate (%) ↓   Error (cm) ↓   Latency (s) ↓
steps used by the autoregressive denoiser. Using extremely few                  Dataset (HumanML3D retarget)                     0.739     0.000       7.92           0.00              -
diffusion steps (e.g., 1 or 2) leads to significantly worse generation          Dataset (Our retarget)                           0.732     0.011       6.87           0.00              -
                                                                                                                             Without optimization
quality and constraint adherence. Increasing diffusion steps provides
                                                                                MaskControl∗ [Pinyoanuntapong et al. 2025]      0.760     0.050        7.27          46.18           0.46
slight gains in FID, R-Precision, and constraint accuracy. However,             ARDY (Ours)∗                                    0.729     0.044        6.28          4.15            0.15
our few-step models still achieve highly competitive performance,                                                             With optimization
                                                                                MaskControl [Pinyoanuntapong et al. 2025]       0.758     0.047        7.87          0.45            68.65
demonstrating the robustness of the learned hybrid representation               ARDY (Ours) Opt                                 0.721     0.088        5.87          0.30            9.25
for efficient high-quality motion synthesis.

   Tokenizer Patch Size. We also evaluate the effect of the tokenizer          6.1       Experiment Setting
patch size. Using a minimum patch size of a single frame leads to
                                                                                  HumanML3D Dataset. This public dataset contains around 30
faster learning in the early stages, but causes training instability
                                                                               hours of motion data with corresponding text descriptions. In our
later on, resulting in significantly worse overall performance in the
                                                                               experiments, we exclude the HumanAct12 [Guo et al. 2020] subset
end. Conversely, using a larger patch size of 8 slightly improves
                                                                               of HumanML3D due to the absence of native joint rotation data and
the FID and R-precision metrics, but at the cost of worse skating
                                                                               the severe motion artifacts introduced by the original preprocessing.
performance and constraint accuracy. This trade-off occurs because
                                                                               During data processing, we preserve the original SMPL [Loper et al.
compressing more frames into a single token causes a greater loss
                                                                               2015] joint rotations in the retargeting step, unlike the original
of fine-grained pose details within each patch.
                                                                               HumanML3D pipeline, which discards native joint rotations. This
   Tokenizer Latent Space Capacity. We evaluate tokenizers with                makes our processed data compatible with real-time animation,
varying latent space capacities. The capacity of a Finite Scalar Quan-         since we can directly animate the body model with generated joint
tization (FSQ) latent space is determined jointly by the number of             rotations instead of going through an expensive inverse kinematics
discrete quantization levels and the number of latent dimensions.              post-process using generated joint positions.
By default, we use an FSQ configuration with 64 quantization levels                Evaluation Metrics. We adopt the evaluation benchmark from
and 128 dimensions, denoted as FSQ 64-128. While performance                   prior work [Guo et al. 2022; Pinyoanuntapong et al. 2025] to assess
is relatively similar across configurations, there are some notable            various aspects of the generated motion. To evaluate text-following,
differences. Using FSQ 16-32 with a smaller latent capacity yields             we report the Top-3 R-precision. Motion quality is measured via
slightly better FID and R-precision metrics under the limited train-           Fréchet Inception Distance (FID), which indicates similarity to the
ing budget of 1 million iterations, but it degrades performance on             ground-truth distribution, and the foot skating ratio, which quan-
end-effector joint constraints and full-body errors. This trade-off            tifies the frequency of detected foot skating frames. To assess spatial
arises because a smaller latent space lacks the capacity to represent          control accuracy, we calculate the mean joint error for the con-
fine-grained motion details accurately. On the other hand, expand-             strained goal joint positions. We utilize the original HumanML3D
ing the number of dimensions to 256 slows model convergence and                evaluator models, which were trained on the original processed
does not provide performance gains within the same train budget.               HumanML3D data, to calculate FID and R-precision metrics. As a
                                                                               result, our method is slightly disadvantaged on these metrics. Addi-
   Tokenizer Type. We experiment with several tokenizer architec-              tionally, we report the motion generation latency for each method,
tures, including Variational Autoencoders (VAE) and Finite Scalar              measured on a single NVIDIA A100-SXM4-80GB GPU.
Quantization (FSQ). For the VAE variant, we applied a KL-divergence
loss with weight of 1 × 10−6 to regularize the latent distribution.            6.2       Offline Model Comparison
Our results indicate that all tokenizer variants perform comparably
                                                                               We first compare to MaskControl [Pinyoanuntapong et al. 2025], a
to a vanilla autoencoder. However, the vanilla autoencoder suffers
                                                                               SOTA offline motion generation model that specializes in accurate
from severe training instability and diverges when trained with
                                                                               joint controls. Following the protocol in MaskControl, we evaluate
longer horizons, such as 40 frames. In contrast, the FSQ tokenizer
                                                                               the model’s ability to satisfy arbitrary joint position constraints
demonstrates superior training stability over the vanilla autoen-
                                                                               at any given frame. We first compare our raw generation results
coder baseline, leading us to adopt FSQ as our default choice.
                                                                               against MaskControl with its test-time optimization module dis-
                                                                               abled (denoted as MaskControl* ). Subsequently, we apply a similar
6   Benchmark Evaluation                                                       test-time optimization to our predicted hybrid motion to minimize
Lastly, we evaluate ARDY against both offline and online state-of-             joint errors. We then compare these refined results against the full
the-art baselines for text+constraints-conditioned generation on the           MaskControl pipeline. As shown in Tab. 4, ARDY achieves competi-
standard HumanML3D [Guo et al. 2022] dataset. For these exper-                 tive text-following (on par with ground truth R-prec) and motion
iments, our model is trained with a 40-frame generation horizon                quality while demonstrating a lower foot skating ratio. Notably,
using 10 diffusion steps and a vanilla autoencoder tokenizer.                  compared to the raw MaskControl output before optimization, our

                                                                                             ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:12    •   Zhao et al.


Table 5. Autoregressive Text and Constraint Control Comparison.                                  Table 6. Perceptual Study Results. We report the percentage of human
evaluation results of text-conditioned autoregressive motion generation                          preferences comparing our method against DiP across three criteria. Our
with in-horizon and out-of-horizon sparse joint goals on HumanML3D. ↑                            approach is consistently preferred over DiP, with a significant margin in
denotes higher values are better; ↓ denotes lower values are better.                             motion quality, semantic alignment, and goal accuracy.

 Method                         R-Prec. ↑   FID ↓   Skate (%) ↓   Error (cm) ↓   Latency (s) ↓                              Ours (%)    Tie (%)   DiP [Tevet et al. 2025] (%)
 Dataset (HumanML3D retarget)     0.711     0.000        8.53         0.00             -
 Dataset (Our retarget)           0.711     0.010        7.00         0.00             -
                                                                                                     Motion Quality           65.8       25.0                 9.2
                                                                                                     Semantic Alignment       67.5       25.0                 7.5
                                      In-horizon goals
                                                                                                     Goal Accuracy            64.6       31.2                 4.2
 DiP [Tevet et al. 2025]         0.609      0.967        12.29       9.20            0.15
 ARDY (Ours)                     0.690      0.092        7.07        2.48            0.15
                                    Out-of-horizon goals
 DiP [Tevet et al. 2025]         0.599      1.453        11.07       17.64           0.15        present a real-time demonstration of interactive and instructable mo-
 ARDY (Ours)                     0.684      0.100        7.63        2.92            0.15
                                                                                                 tion generation, underscoring the potential of generative models for
                                                                                                 future animation systems. We validate our architectural decisions
method yields significantly lower spatial control errors, indicating                             through extensive ablation studies on the large-scale, studio-quality
a stronger underlying generative prior.                                                          Bones Rigplay dataset. Furthermore, experiments on the public Hu-
                                                                                                 manML3D benchmark demonstrate that ARDY outperforms existing
6.3      Autoregressive Model Comparison                                                         methods in terms of both motion fidelity and control accuracy.
Next, we compare ARDY to the closely related model DiP [Tevet                                       Limitations. While ARDY demonstrates a promising system for
et al. 2025], an autoregressive motion diffusion model. For the au-                              interactive human motion generation, several aspects of the design
toregressive model comparison, we evaluate constraints satisfac-                                 remain open for future research improvement. First, ARDY explic-
tion by sampling goal joints using two distinct schemes. The first                               itly utilizes all past motion frames as the history context during
scheme, termed in-horizon goals, follows the original DiP setting by                             autoregressive generation, which can be inefficient for extremely
sampling one goal joint at the final frame of each autoregressive                                long-horizon tasks. Exploring more efficient, structured memory
generation window. This scheme necessitates a goal input every                                   representations and update mechanisms is an important future di-
2 seconds, which is often impractical for applications relying on                                rection. Second, as a diffusion model, ARDY relies on a multi-step
sparser control signals. The second scheme, out-of-horizon goals,                                iterative generation process, which can be computationally demand-
involves sampling a single final goal joint at the very end of a long                            ing. This could potentially be further accelerated by combining our
sequence which is beyond the initial autoregressive generation win-                              approach with recent advances in shortcut diffusion models [Geng
dow. This configuration creates a challenging scenario requiring                                 et al. 2025; Lu and Song 2025]. Third, ARDY is a purely kinematic
long-horizon planning, a task that the DiP system fails to handle                                model and lacks awareness of physical dynamics. Consequently, ar-
effectively. Following the implementation of DiP, we sample the                                  tifacts such as foot skating and jittering can sometimes be observed
goal joints from the pelvis, wrists, and feet. We set the test sequence                          in the generated motions. A crucial future direction is to integrate
length to 9 seconds and provide 1 second of ground truth motion as                               physics modelling into ARDY, proposing a unified generative model
initial history to adapt to the original DiP implementation.                                     capable of predicting both the kinematics and dynamics of human
   As presented in Tab. 5, our approach surpasses DiP in both in-                                motion, which is essential for physics-critical applications.
horizon and out-of-horizon scenarios. Notably, DiP exhibits a sharp
increase in joint error under the out-of-horizon setting, indicating                             8   Acknowledgments
its limitation for long-term planning. In contrast, our method ef-
                                                                                                 We would like to thank Edy Lim, Eugene Jeong, Sam Wu, Ehsan
fectively resolves these long-context constraints, maintaining high
                                                                                                 Hassani, Michael Huang, and Jin-Bey Yu for their help with data
accuracy even when goals are placed far into the future. Further-
                                                                                                 processing and cleaning, and Cyrus Hogg, Simon Yuen, Lindsey
more, to ensure our quantitative gains translate to actual human
                                                                                                 Pavao, Jenna Diamond, Rizwan Khan, Samantha Shinagawa, and
perception, we conduct a side-by-side perceptual study comparing
                                                                                                 Akanksha Shukla for their efforts on data acquisition and labeling.
the two methods on motion quality, semantic alignment, and joint
                                                                                                 We also thank the anonymous reviewers for their valuable feedback.
goal accuracy for out-of-horizon goals. Participants are instructed
to vote for the better result or indicate a tie. Across 240 pairwise
                                                                                                 References
human comparisons (Tab. 6), our approach ARDY is strongly and                                    AI@Meta. 2024. Llama 3 Model Card. (2024). https://github.com/meta-llama/llama3/
consistently preferred over DiP, confirming that the numerical im-                                  blob/main/MODEL_CARD.md
provements in Tab. 5 reflect genuine qualitative gains.                                          German Barquero, Sergio Escalera, and Cristina Palmero. 2024. Seamless Human
                                                                                                    Motion Composition with Blended Positional Encodings. (2024).
                                                                                                 Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau,
7       Discussion                                                                                  Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large Language Models
                                                                                                    Are Secretly Powerful Text Encoders. In First Conference on Language Modeling.
We propose ARDY, an autoregressive motion diffusion model that en-                                  https://openreview.net/forum?id=IW1PR7vEBf
ables interactive and controllable human motion generation. ARDY                                 Bones Studio. 2026. AI Datasets for Machine Learning and Motion Capture. https:
natively supports online text prompting and flexible kinematic                                      //bones.studio/datasets. Accessed: 2026.
                                                                                                 Zhi Cen, Huaijin Pi, Sida Peng, Qing Shuai, Yujun Shen, Hujun Bao, Xiaowei Zhou,
goal constraints tailored to interactive applications, including long-                              and Ruizhen Hu. 2025. Ready-to-React: Online Reaction Policy for Two-Character
horizon goals that extend beyond a single generation window. We                                     Interaction Generation. In ICLR.


ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
                                                                 ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation         •   86:13


Rui Chen, Mingyi Shi, Shaoli Huang, Ping Tan, Taku Komura, and Xuelin Chen. 2024.           Chuqiao Li, Julian Chibane, Yannan He, Naama Pearl, Andreas Geiger, and Gerard Pons-
   Taming Diffusion Probabilistic Models for Character Control. In ACM SIGGRAPH                Moll. 2025. Unimotion: Unifying 3d human motion synthesis and understanding. In
   2024 Conference Papers (Denver, CO, USA) (SIGGRAPH ’24). Association for Com-               2025 International Conference on 3D Vision (3DV). IEEE, 240–249.
   puting Machinery, New York, NY, USA. doi:10.1145/3641519.3657440                         Qiayuan Liao, Takara E Truong, Xiaoyu Huang, Guy Tevet, Koushil Sreenath, and
Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. 2023.              C Karen Liu. 2025. Beyondmimic: From motion tracking to versatile humanoid
   Executing your Commands via Motion Diffusion in Latent Space. In Proceedings of             control via guided diffusion. arXiv preprint arXiv:2508.08241 (2025).
   the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18000–18010.         Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. 2020. Character
Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang.                 controllers using motion vaes. ACM Transactions on Graphics (TOG) 39, 4 (2020),
   2025. Motionlcm: Real-time controllable motion generation via latent consistency            40–1.
   model. In ECCV. 390–408.                                                                 Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J.
Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A Alemi, Roman Novak,                 Black. 2015. SMPL: A Skinned Multi-Person Linear Model. ACM Trans. Graphics
   Peter J Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee,       (Proc. SIGGRAPH Asia) (2015).
   et al. 2024. Scaling exponents across parameterizations and optimizers. International    Cheng Lu and Yang Song. 2025. Simplifying, Stabilizing and Scaling Continuous-
   Conference on Machine Learning (2024).                                                      time Consistency Models. In The Thirteenth International Conference on Learning
Ke Fan, Shunlin Lu, Minyue Dai, Runyi Yu, Lixing Xiao, Zhiyang Dou, Junting Dong,              Representations.
   Lizhuang Ma, and Jingbo Wang. 2025. Go to Zero: Towards Zero-shot Motion                 Shunlin Lu, Jingbo Wang, Zeyu Lu, Ling-Hao Chen, Wenxun Dai, Junting Dong, Zhiyang
   Generation with Million-scale Data. In Proceedings of the IEEE/CVF International            Dou, Bo Dai, and Ruimao Zhang. 2025. Scamo: Exploring the scaling law in autore-
   Conference on Computer Vision (ICCV). arXiv:2507.07095 [cs.CV] https://arxiv.org/           gressive motion generation model. In Proceedings of the Computer Vision and Pattern
   abs/2507.07095                                                                              Recognition Conference. 27872–27882.
Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. 2015. Recur-         Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and
   rent network models for human dynamics. In Proceedings of the IEEE international            Weipeng Xu. 2023. Universal humanoid motion representations for physics-based
   conference on computer vision. 4346–4354.                                                   control. arXiv preprint arXiv:2310.04582 (2023).
Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. 2025.           Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Castañeda,
   Mean Flows for One-step Generative Modeling. In The Thirty-ninth Annual Confer-             Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Xingye Da, Runyu Ding, Cyrus
   ence on Neural Information Processing Systems.                                              Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Zi
Ross Girshick. 2015. Fast R-CNN. In International Conference on Computer Vision (ICCV).        Wang, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Fan, and Yuke Zhu.
Ruiyu Gou, Michiel van de Panne, and Daniel Holden. 2025. Control Operators for                2025. SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body
   Interactive Character Animation. ACM Transactions on Graphics (TOG) (2025).                 Control. arXiv preprint arXiv:2511.07820 (2025).
Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. 2024.                   Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang. 2025. Re-
   Momask: Generative masked modeling of 3d human motions. In Proceedings of the               thinking Diffusion for Text-Driven Human Motion Generation: Redundant Repre-
   IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1900–1910.                  sentations, Evaluation, and Masked Autoregression. In Proceedings of the Computer
Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. 2022.            Vision and Pattern Recognition Conference. 27859–27871.
   Generating Diverse and Natural 3D Human Motions From Text. In Proceedings of the         Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. 2023. Finite
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 5152–5161.           scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505 (2023).
Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun               Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. 2022. Ase:
   Gong, and Li Cheng. 2020. Action2motion: Conditioned generation of 3d human                 Large-scale reusable adversarial skill embeddings for physically simulated characters.
   motions. In Proceedings of the 28th ACM international conference on multimedia.             ACM Transactions On Graphics (TOG) 41, 4 (2022), 1–17.
   2021–2029.                                                                               Mathis Petrovich, Michael J. Black, and Gül Varol. 2022. TEMOS: Generating diverse
Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and              human motions from textual descriptions. In European Conference on Computer
   Michael J Black. 2021. Stochastic scene-aware motion prediction. In Proceedings of          Vision (ECCV).
   the IEEE/CVF International Conference on Computer Vision. 11374–11384.                   Mathis Petrovich, Michael J. Black, and Gül Varol. 2023. TMR: Text-to-Motion Retrieval
Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi              Using Contrastive 3D Human Motion Synthesis. In International Conference on
   Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, et al. 2025. Asap: Aligning simulation         Computer Vision (ICCV).
   and real-world physics for learning agile humanoid whole-body skills. arXiv preprint     Mathis Petrovich, Or Litany, Umar Iqbal, Michael J. Black, Gül Varol, Xue Bin Peng,
   arXiv:2502.01143 (2025).                                                                    and Davis Rempe. 2024. Multi-Track Timeline Control for Text-Driven 3D Human
Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic             Motion Generation. In CVPR Workshop on Human Motion Generation.
   models. Advances in neural information processing systems 33 (2020), 6840–6851.          Ekkasit Pinyoanuntapong, Muhammad Saleem, Korrawe Karunratanakul, Pu Wang,
Jonathan Ho and Tim Salimans. 2021. Classifier-Free Diffusion Guidance. In NeurIPS             Hongfei Xue, Chen Chen, Chuan Guo, Junli Cao, Jian Ren, and Sergey Tulyakov.
   2021 Workshop on Deep Generative Models and Downstream Applications.                        2025. MaskControl: Spatio-Temporal Control for Masked Motion Synthesis. In
Daniel Holden, Oussama Kanoun, Maksym Perepichka, and Tiberiu Popa. 2020. Learned              Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
   motion matching. ACM Transactions on Graphics (ToG) (2020).                                 9955–9965.
Daniel Holden, Taku Komura, and Jun Saito. 2017. Phase-functioned neural networks           Ekkasit Pinyoanuntapong, Muhammad Usama Saleem, Pu Wang, Minwoo Lee, Srijan
   for character control. ACM Transactions on Graphics (TOG) 36, 4 (2017), 1–13.               Das, and Chen Chen. 2024a. Bamm: Bidirectional autoregressive motion model. In
Xiaoyu Huang, Takara Truong, Yunbo Zhang, Fangzhou Yu, Jean Pierre Sleiman, Jessica            European Conference on Computer Vision. Springer, 172–190.
   Hodgins, Koushil Sreenath, and Farbod Farshidian. 2025. Diffuse-cloc: Guided             Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. 2024b. Mmm:
   diffusion for physics-based character look-ahead control. ACM Transactions on               Generative masked motion model. In Proceedings of the IEEE/CVF Conference on
   Graphics (TOG) 44, 4 (2025), 1–12.                                                          Computer Vision and Pattern Recognition. 1546–1555.
Kaiyang Ji, Ye Shi, Zichen Jin, Kangyi Chen, Lan Xu, Yuexin Ma, Jingyi Yu, and Jingya       Matthias Plappert, Christian Mandery, and Tamim Asfour. 2016. The kit motion-
   Wang. 2025. Towards immersive human-x interaction: A real-time framework for                language dataset. Big data 4, 4 (2016), 236–252.
   physically plausible motion synthesis. In Proceedings of the IEEE/CVF International      Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving
   Conference on Computer Vision. 10173–10183.                                                 language understanding by generative pre-training. (2018).
Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. 2024a. Motiongpt:          Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, and
   Human motion as a foreign language. Advances in Neural Information Processing               Leonidas J Guibas. 2021. Humor: 3d human motion model for robust pose esti-
   Systems 36 (2024).                                                                          mation. In Proceedings of the IEEE/CVF international conference on computer vision.
Nan Jiang, Zimo He, Zi Wang, Hongjie Li, Yixin Chen, Siyuan Huang, and Yixin Zhu.              11488–11499.
   2024b. Autonomous character-scene interaction synthesis from text instruction. In        Davis Rempe, Zhengyi Luo, Xue Bin Peng, Ye Yuan, Kris Kitani, Karsten Kreis, Sanja
   SIGGRAPH Asia 2024 Conference Papers. 1–11.                                                 Fidler, and Or Litany. 2023. Trace and pace: Controllable pedestrian animation via
Korrawe Karunratanakul, Konpat Preechakul, Emre Aksan, Thabo Beeler, Supasorn                  guided trajectory diffusion. In Proceedings of the IEEE/CVF Conference on Computer
   Suwajanakorn, and Siyu Tang. 2024. Optimizing diffusion noise can serve as uni-             Vision and Pattern Recognition. 13756–13766.
   versal motion priors. In Proceedings of the IEEE/CVF Conference on Computer Vision       Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, Yifeng Jiang,
   and Pattern Recognition. 1334–1345.                                                         Tingwu Wang, Umar Iqbal, David Minor, Michael de Ruyter, Jiefeng Li, Chen Tessler,
Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang.               Edy Lim, Eugene Jeong, Sam Wu, Ehsan Hassani, Michael Huang, Jin-Bey Yu,
   2023. Guided motion diffusion for controllable human motion synthesis. In Proceed-          Chaeyeon Chung, Lina Song, Olivier Dionne, Jan Kautz, Simon Yuen, and Sanja Fidler.
   ings of the IEEE/CVF International Conference on Computer Vision. 2151–2162.                2026. Kimodo: Scaling Controllable Human Motion Generation. arXiv:2603.15546
Diederik P Kingma and Max Welling. 2014. Auto-Encoding Variational Bayes. In                   (2026).
   International Conference on Learning Representations.



                                                                                                        ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
86:14   •   Zhao et al.


Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Xiao Ma, Liang Pan, and Ziwei Liu. 2023.               International Conference on Learning Representations.
   InsActor: Instruction-driven Physics-based Characters. NeurIPS (2023).                 Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang,
Cohan Setareh, Guy Tevet, Daniele Reda, Xue Bin Peng, and Michiel van de Panne.               Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, and Angjoo Kanazawa.
   2024. Flexible Motion In-betweening with Diffusion Models. (2024).                         2025. Viser: Imperative, web-based 3d visualization in python. arXiv preprint
Yi Shi, Jingbo Wang, Xuekun Jiang, Bingkun Lin, Bo Dai, and Xue Bin Peng. 2024.               arXiv:2507.22885 (2025).
   Interactive Character Control with Auto-Regressive Motion Diffusion Models. ACM        Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei
   Trans. Graph. 43 (jul 2024).                                                               Zhao, Hongtao Lu, and Xi Shen. 2023. T2M-GPT: Generating Human Motion from
Sebastian Starke, Ian Mason, and Taku Komura. 2022. Deepphase: Periodic autoencoders         Textual Descriptions with Discrete Representations. In Proceedings of the IEEE/CVF
   for learning motion phase manifolds. ACM Transactions on Graphics (ToG) 41, 4              Conference on Computer Vision and Pattern Recognition (CVPR).
   (2022), 1–13.                                                                          Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang,
Sebastian Starke, He Zhang, Taku Komura, and Jun Saito. 2019. Neural state machine            and Ziwei Liu. 2024a. Motiondiffuse: Text-driven human motion generation with
   for character-scene interactions. ACM Transactions on Graphics 38, 6 (2019), 178.          diffusion model. IEEE transactions on pattern analysis and machine intelligence 46, 6
Graham W Taylor, Geoffrey E Hinton, and Sam Roweis. 2006. Modeling human motion              (2024), 4115–4128.
   using binary latent variables. Advances in neural information processing systems 19    Yan Zhang, Yao Feng, Alpár Cseke, Nitin Saini, Nathan Bajandas, Nicolas Heron, and
   (2006).                                                                                    Michael J. Black. 2025. PRIMAL: Physically Reactive and Interactive Motor Model
Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. 2024. Masked-          for Avatar Learning. In Proceedings of the IEEE/CVF International Conference on
   Mimic: Unified Physics-Based Character Control Through Masked Motion Inpaint-              Computer Vision (ICCV).
   ing. ACM Transactions on Graphics (TOG) (2024).                                        Yan Zhang and Siyu Tang. 2022. The wanderings of odysseus in 3d scenes. In Proceedings
Guy Tevet, Sigal Raab, Setareh Cohan, Daniele Reda, Zhengyi Luo, Xue Bin Peng,                of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20481–20491.
   Amit Haim Bermano, and Michiel van de Panne. 2025. CLoSD: Closing the Loop             Zihan Zhang, Richard Liu, Rana Hanocka, and Kfir Aberman. 2024b. Tedi: Temporally-
   between Simulation and Diffusion for multi-task character control. In The Thirteenth       entangled diffusion for long-term motion synthesis. In ACM SIGGRAPH 2024 Con-
   International Conference on Learning Representations. https://openreview.net/forum?        ference Papers. 1–11.
   id=pZISppZSTv                                                                          Kaifeng Zhao, Gen Li, and Siyu Tang. 2025a. DartControl: A Diffusion-Based Autore-
Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim              gressive Motion Model for Real-Time Text-Driven Motion Control. In The Thirteenth
   Bermano. 2023. Human Motion Diffusion Model. In The Eleventh International Con-            International Conference on Learning Representations (ICLR).
   ference on Learning Representations. https://openreview.net/forum?id=SJ1kSyO2jwu       Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, and Siyu Tang. 2023. Synthe-
Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, and                     sizing diverse human motions in 3d indoor scenes. In Proceedings of the IEEE/CVF
   Lingjie Liu. 2024. Tlcontrol: Trajectory and language control for human motion             international conference on computer vision. 14738–14749.
   synthesis. In European Conference on Computer Vision. Springer, 37–54.                 Siheng Zhao, Yanjie Ze, Yue Wang, C Karen Liu, Pieter Abbeel, Guanya Shi, and Rocky
Yan Wu, Korrawe Karunratanakul, Zhengyi Luo, and Siyu Tang. 2025. UniPhys: Unified            Duan. 2025b. ResMimic: From General Motion Tracking to Humanoid Whole-body
   Planner and Controller with Diffusion for Flexible Physics-Based Character Control.        Loco-Manipulation via Residual Learning. arXiv preprint arXiv:2510.05070 (2025).
   arXiv preprint arXiv:2504.12540 (2025).                                                Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia
Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng,             Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. 2024. Emdm:
   Xiaowei Zhou, Sida Peng, and Jingbo Wang. 2025. MotionStreamer: Streaming                  Efficient motion diffusion model for fast and high-quality motion generation. In
   Motion Generation via Diffusion-based Autoregressive Model in Causal Latent                European Conference on Computer Vision. Springer, 18–38.
   Space. arXiv preprint arXiv:2503.15451 (2025).                                         Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. 2019. On the continuity
Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. 2024. OmniCon-            of rotation representations in neural networks. In Proceedings of the IEEE/CVF
   trol: Control Any Joint at Any Time for Human Motion Generation. In The Twelfth            conference on computer vision and pattern recognition. 5745–5753.




ACM Trans. Graph., Vol. 45, No. 4, Article 86. Publication date: July 2026.
