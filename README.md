# multi-task-learning-model
基于keras的多任务模型，包括经典的阿里的esmm模型以及google的mmoe模型。以及业务迭代模型esmm_v3已经在公司业务上线。
本代码在data文件中提供了示例样本，为脱敏后的业务数据。所有模型使用相同对数据。
data的具体含义见data内的readme.md文档
## ESMM架构
《Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate》
### esmm_base：论文复现
曝光——点击——支付；其中点击和支付是已知标签
结构如下所示：
![image](https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm.png)

### 业务迭代版 esmm_v2
曝光——点击——申请——核验——激活，更多的中间目标。结构如下：
![image](https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v2.png)

### 业务迭代版 esmm_v3
修改成了新的多目标模型结构，相比于esmm_v2, 前一步有更多的信息被引入到后一步中，且靠后的网络层能够很好的包含更高级的信息。该结构已申请发明专利。结构如下：
<img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v3.PNG" width="600" height="600"/><br/>

### 业务迭代版 esmm_v4
相比于esmm_v3将前一步所有信息引入(concatenate),该版本加入Adaptive Information Transfer 模块，将t-1步的信息和第t步的信息自动计算权重再融合。
<img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v4.PNG" width="400" height="400"/><br/>

## MMOE
<Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts>
 实现了论文中的mmoe结构,mmoe_v2进行改进，使能够支持多层mmoe结构。
 
![image](https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/mmoe.png)
 

