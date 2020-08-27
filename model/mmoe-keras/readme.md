## mmoe
mmoe.py定义了一层mmoe层，参数：units,number_experts,num_tasks.
但要注意的是只有一层网络
![image](https://github.com/zhangyingerjelly/multi-task-learning-model/blob/master/img/mmoe.png)

## mmoe_v2:
改进了mmoe只有一层的缺点：mmoe_layers = MMoE(units=[16,4],num_experts=8,num_tasks=3)(input_concatenate_layer)
可以看到一个expert可以是多层dense.
