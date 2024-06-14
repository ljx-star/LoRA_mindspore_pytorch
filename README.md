# LoRA: Low-Rank Adaptation of Large Language Models

# 1. 论文介绍  
## 1.1 背景介绍  
-  LoRA模型，全称Low-Rank Adaptation of Large Language Models，是一种用于微调大型语言模型的低秩适应技术。它最初应用于NLP领域，特别是用于微调GPT-3等模型。LoRA通过仅训练低秩矩阵，然后将这些参数注入到原始模型中，从而实现对模型的微调。这种方法不仅减少了计算需求，而且使得训练资源比直接训练原始模型要小得多，因此非常适合在资源有限的环境中使用。
-  这种方法通过在模型的低秩特征空间中进行适配，而不是直接修改模型的全连接层或卷积层，从而实现了更高效和可控的模型微调。LORA模型的核心思想是在模型的特征空间中引入辅助特征，这些辅助特征用于捕捉特定任务或领域的信息。这些辅助特征通过低秩分解与原始特征相结合，从而实现对模型的微调。
  
## 1.2 论文方法  
-  LoRA 对Stable Diffusion模型中最关键的部分进行微小改动：交叉注意力层。这是模型中图像和提示相遇的部分。论文发现，仅微调模型的这一部分就足以实现良好的训练效果。交叉注意力层在下方的Stable Diffusion模型架构中以黄色部分表示。

-   ![image](https://github.com/ljx-star/LoRA_mindspore_pytorch/blob/master/ma-user/%E5%9B%BE%E7%89%871.png)  

-  LoRA 通过在冻结原始权重的同时学习秩分解矩阵对来减少可训练参数的数量。 这大大降低了适用于特定任务的大型语言模型的存储需求，并在部署期间实现了高效的任务切换，所有这些都不会引入推理延迟。 LoRA 还优于其他几种适配方法，包括适配器、前缀调整和微调。使用 RoBERTa （Liu et al.， 2019） base and large 和 DeBERTa （He et al.， 2020） XXL 1.5B 在 GLUE 基准上获得与完全微调相当或优于完全微调的结果，同时仅训练和存储一小部分参数。
-  在 GPT-2 上，LoRA 与完全微调和其他高效调优方法相比都具有优势，在 E2E NLG Challenge、DART 和 WebNLG 上进行了评估。
-  ![image](https://github.com/ljx-star/LoRA_mindspore_pytorch/blob/master/ma-user/%E5%9B%BE%E7%89%872.png)  

# 3.基于pytorch框架的论文复现和小规模迁移
## 3.1 环境准备与配置（TODO）
-  创建环境：  
```     
conda create -n lora python=3.8    
```   
loralib:  
```python   
pip install loralib  
#Alternatively    
#pip install git+https://github.com/microsoft/LoRA    
```

-  先进行LoRA微调，随后进行测评。  
-  分别基于NLU和NLG进行LoRA微调。    
![image](https://github.com/ljx-star/LoRA_mindspore_pytorch/blob/master/ma-user/%E5%9B%BE%E7%89%873.png)   
![image](https://github.com/ljx-star/LoRA_mindspore_pytorch/blob/master/ma-user/%E5%9B%BE%E7%89%874.png)  

## 3.2 实验设计与复现

### 3.2.1 实验的总体设计


### 3.2.2 数据集与评测流程


### 3.2.3 实验数据与结论


## 3.3 算法核心部分的迁移

- 本部分主要考虑对loralib库中py文件从pytorch架构向mindspore架构的迁移
- microsoft/LoRA仓库的其余部分均可认定为LoRA在针对NLU和NLG任务时在具体数据集上的实验设计部分
  -  一方面，这些环节尽管会使用pytorch库的内容，但实际上与AI模型和微调算法本身的迁移关系有限，且并没有成为python包管理的一部分，因此迁移的价值有限
  -  另一方面，实验数据集实现部分其涉猎内容过多，工作量巨大，用于测试的modelarts线上运行环境使用时间有限
  -  因此主要迁移微调算法计入python包管理系统的部分（loralib）

-  代码迁移与设计文档写作相关内容均主要参照torch与mindspore的映射文档[MindSpore](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)

### `loralib/layer.py`

- 迁移时主要使用的库：

  - torch → mindspore
  - torch.nn → mindspore.nn
  - torch.nn.functional → torch.nn.ops

- 迁移时修改的对象及其属性、方法

- |      | torch                          | mindspore                              | 注释                                                         |
  | ---- | ------------------------------ | -------------------------------------- | ------------------------------------------------------------ |
  | 1.   | torch.nn.Dropout               | mindspore.nn.Dropout                   | 参数的p改为keep_prob，两者之间是和为1的关系，需要转换        |
  | 2.   | torch.nn.Parameter             | mindspore.Parameter                    | 后者有更多的参数，包括**name**(str)与一些并行模式的配置(bool) |
  | 3.   | torch.nn.init.zeros_           | mindspore.common.initializer.Zero      |                                                              |
  | 4.   | torch.nn.init.normal_          | mindspore.common.initializer.Normal    | initializer导致初始化方法有所区别，此外正态分布默认值定义不一样 |
  | 5.   | torch.nn.init.kaiming_uniform_ | mindspore.common.initializer.HeUniform |                                                              |
  | 6. | torch.nn.Embedding     |            mindspore.nn.Embedding                         |              后者支持独热编码                                       |
  | 7. | torch.nn.Embedding.reset_parameter     | ——                                     | 无对应                                                       |
  | 8. | torch.nn.Embedding.train               | ——                                     | 无对应                                                       |
  | 9. | torch.nn.Embedding.forward               | ——                                     | 无对应                                                       |
  | 10. | torch.nn.Linear                | mindspore.nn.Dense                     | 分别默认均匀分布和正态分布，另外后者可以添加激活函数         |
  | 11. | torch.Tensor                   | mindspore.Tensor                       |                                                              |
  | 12. | torch.matmul | mindspore.nn.Matmul |                                                              |
  | 13. | torch.mm | mindspore.ops.MatMul |                                                              |
  | 14. | torch.mul | mindspore.ops.Mul |                                                              |
  | 15. | torch.add | mindspore.ops.Add |                                                              |
  | 16. | torch.nn.Conv1d | mindspore.nn.Conv1d |                                                              |
  | 17. | torch.nn.Conv2d | mindspore.nn.Conv2d | 前者默认padding，有bias，后者反之 |
  | 18. | torch.nn.Conv3d | mindspore.nn.Conv3d |                                                              |
  | 19. | torch.nn.functional.conv2d | mindspore.ops.Conv2D |                                                              |
  | 20. | torch.Tensor.dtype | mindspore.Tensor.dtype |                                                              |

  

### `loralib/utils.py`

- 迁移时主要使用的库：
  - torch → mindspore
  - torch.nn → mindspore.nn
- 迁移时修改的对象及其属性、方法：

- |      | torch                  | mindspore                              | 注释                   |
| ---- | ---------------------- | -------------------------------------- | ---------------------- |
| 1.   | torch.nn.Module       | mindspore.nn.Cell                      | 作为函数声明的参数存在 |
| 2.   | torch.nn.Module.named_parameters | mindspore.nn.Cell.parameters_and_names |                        |
| 3.   | torch.nn.Module.modules          | mindspore.nn.Cell.cells_and_names                  |                        |
| 4.   | torch.nn.Module.state_dict       | mindspore.nn.Cell.parameters_dict                  |                        |



# 4. 基于Mindspore架构的实现与简单评测  
## 4.1 框架介绍  
昇思MindSpore是由华为于2019年8月推出的新一代全场景AI框架，2020年3月28日，华为宣布昇思MindSpore正式开源。昇思MindSpore是一个全场景AI框架，旨在实现易开发、高效执行、全场景统一部署三大目标。
MindSpore是华为推出的一款开源深度学习框架。它提供了一套端到端的开发工具和算法库，旨在简化深度学习模型的开发、训练和部署过程。   
### 与其他深度学习框架相比，MindSpore有以下几个特点： 
#### 1.自动并行计算：  
MindSpore具有自动并行计算的能力，可以根据硬件资源和网络拓扑自动进行并行计算的优化。这意味着你可以专注于模型的设计和开发，而不需要手动处理并行计算。   
#### 2.动态计算图：  
MindSpore采用动态计算图的方式，允许你在运行时动态地构建计算图。与静态计算图相比，动态计算图更加灵活，可以方便地处理可变长度的输入数据。   
### MindSpore的核心概念    
#### 1.张量（Tensor）    
张量是MindSpore中的基本数据类型，可以将其视为多维数组。你可以将张量看作是存储和表示数据的容器，可以是一个标量（0维张量）、向量（1维张量）、矩阵（2维张量）或更高维度的数组。
张量可以存储数字、图像、音频或其他类型的数据。   
#### 2.计算图（Computational Graph）：    
计算图是MindSpore用来描述计算流程的概念。它由一系列的计算节点和数据流组成，表示了模型的计算过程。在MindSpore中，你可以构建一个计算图来定义神经网络的结构和操作。计算图定义了数据从输入到输出的流动路径和相应的计算操作，计算图的优势在于它能够 自动记录 模型中的计算过程，并且可以进行自动微分来进行反向传播和参数更新。它使得深度学习框架能够高效地执行各种复杂的计算操作。
#### 3.操作符（Operator）    
在MindSpore中，操作符是用来进行各种数学和逻辑运算的函数或方法。它们可以用于对张量进行加法、乘法、卷积等操作，构建复杂的计算流程。    
## 4.2 环境准备  
基于华为云服务区，在Modelarts中创建mindspore_2.1.0-cann_6.3.2-py_3.7-euler_2.10.7-aarch64-snt9b环境（Ascend Snt9b+ARM algorithm development and training. MindSpore is preset in the AI engine.）
## 4.3 模型迁移  
在Mindspore中需要替换pytorch的API，例如下表（部分） ，可参考官方文档。

| PyTorch APIs                                                                           | MindSpore APIs                                                                                                                                                          | 说明                                                                                                                     |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [torch.abs](https://pytorch.org/docs/1.5.0/torch.html#torch.abs)                             | [mindspore.ops.Abs](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Abs.html#mindspore.ops.Abs)                                                 |                                                                                                                          |
| [torch.acos](https://pytorch.org/docs/1.5.0/torch.html#torch.acos)                           | [mindspore.ops.ACos](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.ACos.html#mindspore.ops.ACos)                                              |                                                                                                                          |
| [torch.add](https://pytorch.org/docs/1.5.0/torch.html#torch.add)                             | [mindspore.ops.Add](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Add.html#mindspore.ops.Add)                                                 |                                                                                                                          |
| [torch.arange](https://pytorch.org/docs/1.5.0/torch.html#torch.arange)                       | [mindspore.numpy.arange](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.arange.html#mindspore.numpy.arange)                                |                                                                                                                          |
| [torch.argmax](https://pytorch.org/docs/1.5.0/torch.html#torch.argmax)                       | [mindspore.ops.Argmax](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Argmax.html#mindspore.ops.Argmax)                                        |                                                                                                                          |
| [torch.argmin](https://pytorch.org/docs/1.5.0/torch.html#torch.argmin)                       | [mindspore.ops.Argmin](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Argmin.html#mindspore.ops.Argmin)                                        |                                                                                                                          |
| [torch.argsort](https://pytorch.org/docs/1.5.0/torch.html#torch.argsort)                     | [mindspore.ops.Sort](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Sort.html#mindspore.ops.Sort)                                              | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/Sort.html)                |
| [torch.asin](https://pytorch.org/docs/1.5.0/torch.html#torch.asin)                           | [mindspore.ops.Asin](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Asin.html#mindspore.ops.Asin)                                              |                                                                                                                          |
| [torch.atan](https://pytorch.org/docs/1.5.0/torch.html#torch.atan)                           | [mindspore.ops.Atan](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Atan.html#mindspore.ops.Atan)                                              |                                                                                                                          |
| [torch.atan2](https://pytorch.org/docs/1.5.0/torch.html#torch.atan2)                         | [mindspore.ops.Atan2](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Atan2.html#mindspore.ops.Atan2)                                           |                                                                                                                          |
| [torch.bartlett_window](https://pytorch.org/docs/1.5.0/torch.html#torch.bartlett_window)     | [mindspore.numpy.bartlett](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.bartlett.html#mindspore.numpy.bartlett)                          |                                                                                                                          |
| [torch.bincount](https://pytorch.org/docs/1.5.0/torch.html#torch.bincount)                   | [mindspore.numpy.bincount](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.bincount.html#mindspore.numpy.bincount)                          |                                                                                                                          |
| [torch.bitwise_and](https://pytorch.org/docs/1.5.0/torch.html#torch.bitwise_and)             | [mindspore.ops.BitwiseAnd](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.BitwiseAnd.html#mindspore.ops.BitwiseAnd)                            |                                                                                                                          |
| [torch.bitwise_or](https://pytorch.org/docs/1.5.0/torch.html#torch.bitwise_or)               | [mindspore.ops.BitwiseOr](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.BitwiseOr.html#mindspore.ops.BitwiseOr)                               |          
| [torch.flip](https://pytorch.org/docs/1.5.0/torch.html#torch.flip)                           | [mindspore.ops.ReverseV2](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.ReverseV2.html#mindspore.ops.ReverseV2)                               |                                                                                                                          |
| [torch.floor](https://pytorch.org/docs/1.5.0/torch.html#torch.floor)                         | [mindspore.ops.Floor](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Floor.html#mindspore.ops.Floor)                                           |                                                                                                                          |
| [torch.floor_divide](https://pytorch.org/docs/1.5.0/torch.html#torch.floor_divide)           | [mindspore.ops.FloorDiv](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.FloorDiv.html#mindspore.ops.FloorDiv)                                  | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/FloorDiv.html)            |
| [torch.fmod](https://pytorch.org/docs/1.5.0/torch.html#torch.fmod)                           | [mindspore.ops.Mod](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Mod.html#mindspore.ops.Mod)                                                 |                                                                                                                          |
| [torch.from_numpy](https://pytorch.org/docs/1.5.0/torch.html#torch.from_numpy)               | [mindspore.tensor.from_numpy](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Tensor.html?#mindspore.Tensor.from_numpy)                       |                                                                                                                          |
| [torch.full](https://pytorch.org/docs/1.5.0/torch.html#torch.full)                           | [mindspore.numpy.full](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.full.html#mindspore.numpy.full)                                      |                                                                                                                          |
| [torch.full_like](https://pytorch.org/docs/1.5.0/torch.html#torch.full_like)                 | [mindspore.numpy.full_like](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.full_like.html#mindspore.numpy.full_like)                       |                                                                                                                          |
| [torch.gather](https://pytorch.org/docs/1.5.0/torch.html#torch.gather)                       | [mindspore.ops.GatherD](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.GatherD.html#mindspore.ops.GatherD)                                     |                                                                                                                          |
| [torch.ge](https://pytorch.org/docs/1.5.0/torch.html#torch.ge)                               | [mindspore.ops.GreaterEqual](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.GreaterEqual.html#mindspore.ops.GreaterEqual)                      |                                                                                                                          |
| [torch.gt](https://pytorch.org/docs/1.5.0/torch.html#torch.gt)                               | [mindspore.ops.Greater](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Greater.html#mindspore.ops.Greater)                                     |                                                                                                                          |
| [torch.hamming_window](https://pytorch.org/docs/1.5.0/torch.html#torch.hamming_window)       | [mindspore.numpy.hamming](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.hamming.html#mindspore.numpy.hamming)                             |                                                                                                                          |
| [torch.hann_window](https://pytorch.org/docs/1.5.0/torch.html#torch.hann_window)             | [mindspore.numpy.hanning](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.hanning.html#mindspore.numpy.hanning)                             |                                                                                                                          |
| [torch.histc](https://pytorch.org/docs/1.5.0/torch.html#torch.histc)                         | [mindspore.ops.HistogramFixedWidth](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.HistogramFixedWidth.html#mindspore.ops.HistogramFixedWidth) |                                                                                                                          |
| [torch.meshgrid](https://pytorch.org/docs/1.5.0/torch.html#torch.meshgrid)                   | [mindspore.ops.Meshgrid](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Meshgrid.html#mindspore.ops.Meshgrid)                                  |                                                                                                                          |
| [torch.min](https://pytorch.org/docs/1.5.0/torch.html#torch.min)                             | [mindspore.ops.ArgMinWithValue](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.ArgMinWithValue.html#mindspore.ops.ArgMinWithValue)             | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/ArgMinWithValue.html)     |
| [torch.mm](https://pytorch.org/docs/1.5.0/torch.html#torch.mm)                               | [mindspore.ops.MatMul](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul)                                        |                                                                                                                          |
| [torch.mul](https://pytorch.org/docs/1.5.0/torch.html#torch.mul)                             | [mindspore.ops.Mul](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul)                                                 |                                                                                                                          |
| [torch.multinomial](https://pytorch.org/docs/1.5.0/torch.html#torch.multinomial)             | [mindspore.ops.multinomial](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.multinomial.html#mindspore.ops.multinomial)                         |                                                                                                                          |
| [torch.ne](https://pytorch.org/docs/1.5.0/torch.html#torch.ne)                               | [mindspore.ops.NotEqual](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.NotEqual.html#mindspore.ops.NotEqual)                                  |                                                                                                                          |
| [torch.neg](https://pytorch.org/docs/1.5.0/torch.html#torch.neg)                             | [mindspore.ops.Neg](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg)                                                 |                                                                                                                          |
| [torch.norm](https://pytorch.org/docs/1.5.0/torch.html#torch.norm)                           | [mindspore.ops.LpNorm](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.LpNorm.html#mindspore.ops.LpNorm)                                        | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/LpNorm.html)                |
| [torch.numel](https://pytorch.org/docs/1.5.0/torch.html#torch.numel)                         | [mindspore.ops.Size](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Size.html#mindspore.ops.Size)                                              |                                                                                                                          |
| [torch.ones](https://pytorch.org/docs/1.5.0/torch.html#torch.ones)                           | [mindspore.ops.Ones](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Ones.html#mindspore.ops.Ones)                                              |                                                                                                                          |
| [torch.ones_like](https://pytorch.org/docs/1.5.0/torch.html#torch.ones_like)                 | [mindspore.ops.OnesLike](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.OnesLike.html#mindspore.ops.OnesLike)                                  |                                                                                                                          |
| [torch.poisson](https://pytorch.org/docs/1.5.0/torch.html#torch.poisson)                     | [mindspore.ops.poisson](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.poisson.html#mindspore.ops.poisson)                                     |                                                                                                                          |
| [torch.pow](https://pytorch.org/docs/1.5.0/torch.html#torch.pow)                             | [mindspore.ops.Pow](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Pow.html#mindspore.ops.Pow)                                                 |                                                                                                                          |
| [torch.tanh](https://pytorch.org/docs/1.5.0/torch.html#torch.tanh)                           | [mindspore.ops.Tanh](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Tanh.html#mindspore.ops.Tanh)                                              |                                                                                                                          |
| [torch.tensor](https://pytorch.org/docs/1.5.0/torch.html#torch.tensor)                       | [mindspore.Tensor](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)                                              |                                                                                                                          |
| [torch.Tensor](https://pytorch.org/docs/1.5.0/torch.html#torch.Tensor)                       | [mindspore.Tensor](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)                                              |                                                                                                                          |
| [torch.tensordot](https://pytorch.org/docs/1.5.0/torch.html#torch.tensordot)                 | [mindspore.numpy.tensordot](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.tensordot.html)                             |                                                                                                                          |
| [torch.topk](https://pytorch.org/docs/1.5.0/torch.html#torch.topk)                           | [mindspore.ops.TopK](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.TopK.html#mindspore.ops.TopK)                                              | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/TopK.html)                |
| [torch.trace](https://pytorch.org/docs/1.5.0/torch.html#torch.trace)                         | [mindspore.Tensor.trace](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.trace)                                  |                                                                                                                          |
| [torch.transpose](https://pytorch.org/docs/1.5.0/torch.html#torch.transpose)                 | [mindspore.ops.Transpose](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose)                               |                                                                                                                          |
| [torch.unique](https://pytorch.org/docs/1.5.0/torch.html#torch.unique)                       | [mindspore.ops.Unique](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Unique.html#mindspore.ops.Unique)                                        | [差异对比](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_diff/Unique.html)              |
| [torch.unsqueeze](https://pytorch.org/docs/1.5.0/torch.html#torch.unsqueeze)                 | [mindspore.ops.ExpandDims](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.ExpandDims.html#mindspore.ops.ExpandDims)                            |                                                                                                                          |
| [torch.var](https://pytorch.org/docs/1.5.0/torch.html#torch.var)                             | [mindspore.Tensor.var](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.var)                                      |                                                                                                                          |
| [torch.where](https://pytorch.org/docs/1.5.0/torch.html#torch.where)                         | [mindspore.numpy.where](https://mindspore.cn/docs/zh-CN/r1.7/api_python/numpy/mindspore.numpy.where.html#mindspore.numpy.where)                                   |                                                                                                                          |
| [torch.zeros](https://pytorch.org/docs/1.5.0/torch.html#torch.zeros)                         | [mindspore.ops.Zeros](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)                                           |                                                                                                                          |
| [torch.zeros_like](https://pytorch.org/docs/1.5.0/torch.html#torch.zeros_like)               | [mindspore.ops.ZerosLike](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.ZerosLike.html#mindspore.ops.ZerosLike)                               |                                                                                                                          |



另外，需注意nn.Module、linear等。  
## 4.4微调  
基于GPT2和WikiText2数据集进行LoRA微调  
更改train_dataset配置中的dataset_dir设置为处理好的数据路径  
注意单击多卡、多机多卡等的设置    
## 4.5测评  
基于GPT2做LoRA微调后，进行两个方面的评测任务。
### 4.5.1 数据集介绍
-  使用包括WikiText2、2SST-2、IMDB、AG-News、COLA等常用的大模型数据集  
-  WikiText2数据集 
  1、数据来源：Wikitext-2数据集是从维基百科抽取的，包含了维基百科中的文章文本。   
  2、数据内容：Wikitext-2数据集包含维基百科的文章内容，包括各种主题和领域的信息。这些文章是经过预处理和清洗的，以提供干净和可用于训练的文本数据。   
  3、数据规模：Wikitext-2数据集的规模相对较小。它包含了超过2,088,628个词标记（token）的文本，以及其中1,915,997个词标记用于训练，172,430个词标记用于验证和186,716个词标记用于测试。 
- SST-2数据集  
  -  SST-2数据集包含电影评论中的句子和它们情感的人类注释。类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）  
-  IMDB数据集  
  -  IMDB数据集影评数据集，包含5万条IMDB影评，评论的情绪是二元的，专门用于情绪分析。  
-  AG-News数据集  
  -  AG-News数据集包含496,835条来自AG新闻语料库4大类别超过2000个新闻源的新闻文章。  
-  COLA数据集  
  -  COLA数据集来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。  
### 4.5.2 文本生成      
基于WikiText2数据集进行文本生成任务评测。    
### 4.5.3 文本分类    
基于2SST-2、IMDB、AG-News、COLA等常用的文本分类数据集做文本分类测评。评测指标为ACC。      
ACC: COLA-0.693, SST-2-0.808, IMDB-0.834, AG-News-0.841    
