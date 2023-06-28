# guided--landscape-imageGen-diffusion
## 问题介绍 Introduction
图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。
其中，条件图像合成，即输入图片数据，合成真实感图片，在内容生成与图片编辑领域有广泛应用。一种条件图像合成的方式是，用两张图片作为输入，经过处理转换后生成一张新的图片，其中一张输入为语义分割图片（称为mask图），指示生成图片（称为gen图）的语义信息；另一张输入为参考风格图片（称为ref图），从色调等方面指示gen图的风格信息

mage generation tasks have been very application scenario computer vision tasks, and there are still many challenges in generating meaningful, high-quality images from semantic segmentation graphs, such as ensuring the authenticity, clarity, diversity, and aesthetics of the generated images. Among them, conditional image synthesis, i.e., inputting image data and synthesizing realistic images, is widely used in the field of content generation and image editing. One way of conditional image synthesis is to use two images as input, which are processed and transformed to generate a new image, where one input is a semantic segmentation image (called mask graph) indicating the semantic information of the generated image (called gen graph), and the other input is a reference style image (called ref graph) indicating the style information of the gen graph in terms of hue and other aspects

参考(Reference): https://www.educoder.net/competitions/Jittor-4

## 训练数据 Training data
高清（宽512、高384）的风景图片imgs，和它们的语义分割图labels。其中 label 是值在 0~28 的灰度图。

HD (512 wide, 384 high) landscape images imgs, and their semantic segmentation maps labels. where label is a grayscale map with values from 0 to 28.

## 网络架构 Architecture
<img src="Architecture-diagram.svg">

## 训练 Training
环境：分别(stage)在RTX4090 24G, Tesla V100 32G，Tesla A100 80G

Environment: RTX4090 24G, Tesla V100 32G, Tesla A100 80G respectively

训练时间:在RTX4090,V100,A100, 训练一轮vae约1.5-2h, 训练一轮unet约2-2.5h

Training time:In RTX4090,V100,A100, training vae one epoch about 1.5-2h, training unet one epoch about 2-2.5h

Training commands: <code>python3 MainTrain.py</code>

Testing commands(#TODO): <code>python3 MainTest.py</code>

## 注意 Note
Code is not completed

1. stage1的vae训练第二轮突然出现崩塌<br>
In stage1, the second round of vae training suddenly emerged mode collapse
2. 在stage3, vae训练正常,但是unet模式崩塌<br>
In stage3, vae training was normal, but the unet mode collapsed
3. 在stage2, unet正常训练,固定stage1所训练的vae
In stage2, unet trained normally, fixed the vae trained in stage1

## TODO
1. 增加学习率调整<br>
Adding modules for learning rate adjustment
2. 将生成的Gen连续超分<br>
Apply continuous super-resolution to the generated Gen
3. 使用低阶矩阵秩分解，将Unet的attention层M\*N的矩阵分解为[M\*d] * [d\*N],使用语义分割信息再次训练加强分割控制性<br>
Using ControlNet with LoRA
