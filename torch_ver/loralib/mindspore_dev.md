
## loralib从torch框架向mindspore迁移的代码细节

代码迁移与设计文档写作相关内容均主要参照torch与mindspore的映射文档[MindSpore](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)

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


### 其余部分

- 其余部分均为LoRA在针对NLU和NLG任务时在具体数据集上的实验设计部分
- 尽管会使用pytorch库的内容，但实际上与AI模型和微调算法本身的迁移关系有限
- 实验数据集实现部分其涉猎内容过多，工作量巨大，因此主要迁移AI模型和微调算法部分
