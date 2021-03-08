# multi-task-learning-model
基于keras的多任务模型，包括经典的阿里的esmm模型以及google的mmoe模型。以及业务迭代模型esmm_v3已经在公司业务上线。
本代码在data文件中提供了示例样本，为脱敏后的业务数据。所有模型使用相同对数据。
data的具体含义见data内的readme.md文档
## ESMM架构
《Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate》
### esmm_base：论文复现
曝光——点击——支付；其中点击和支付是已知标签
结构如下所示：
<div align=center><img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm.png" width="500" height="500" /></div>
### 业务迭代版 esmm_v2
曝光——点击——申请——核验——激活，更多的中间目标。结构如下，每个圆圈表示已知标签，可计算loss：
<div align=center><img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v2.png" width="500" height="500" /></div>
### 业务迭代版 esmm_v3
曝光——点击——申请——核验——激活。修改成了新的多目标模型结构，相比于esmm_v2, 前一步有更多的信息被引入到后一步中，且靠后的网络层能够很好的包含更高级的信息。该结构已申请发明专利。结构如下：
<div align=center><img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v3.PNG" width="600" height="600" /></div>

### 业务迭代版 esmm_v4
相比于esmm_v3将前一步所有信息引入(concatenate),该版本加入Adaptive Information Transfer 模块，将t-1步的信息和第t步的信息自动计算权重再融合。  
<img src="https://latex.codecogs.com/svg.latex?z_{t}=\sum_{u\epsilon&space;\left&space;\{&space;p_{t-1},q_{t}&space;\right&space;\}}^{}&space;w_{u}&space;h_{1}\left&space;(&space;u&space;\right&space;)" title="z_{t}=\sum_{u\epsilon \left \{ p_{t-1},q_{t} \right \}}^{} w_{u} h_{1}\left ( u \right )" />

<img src="https://latex.codecogs.com/svg.latex?w_{u}=softmax\left&space;(&space;\widehat{w_{u}}&space;\right&space;)" title="w_{u}=softmax\left ( \widehat{w_{u}} \right )" />  

<img src="https://latex.codecogs.com/svg.latex?\widehat{w_{u}}=\frac{\left&space;\langle&space;h_{2}(u),h_{3}(3)&space;\right&space;\rangle}{\sqrt{k}}" title="\widehat{w_{u}}=\frac{\left \langle h_{2}(u),h_{3}(3) \right \rangle}{\sqrt{k}}" />  

<div align=center><img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/esmm_v4.PNG" width="400" height="400"/></div>


## MMOE
<Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts>
 实现了论文中的mmoe结构,mmoe_v2进行改进，使能够支持多层mmoe结构。
 <div align=center><img src="https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/mmoe.png" width="600" height="500" /></div>

 

