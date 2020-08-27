## 数据说明
在 sample_train.csv  sample_valid.csv  sample_test.csv数据集中包含了采样后的示例数据。
可使用pd.read_csv读取文件为dataframe.
### label:共四个，对应esmm_v2的图
'cbc_is_click' 'cbc_is apply' 'cbc_is_credit' 'cbc_is_activate_in_t14'
### input:
info_v4.csv 表示模型要用到的输入特征名称，后面的k\d\c代表该特征数据在最初的类型。数据的预处理在spark中已经完成。我们的数据集是处理以后的。
预处理包含了对类别特征的编码、对连续型变量的分桶等。

### embedding_number_v4.json:
输入特征对应的类别数量，为了在embedding层中使用。
