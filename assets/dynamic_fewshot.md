# 对Dynamic Few-shot的支持

- 本框架支持了这篇工作提出的动态少样本，即在训练集足够大时对每个题目定制地提供少样本示例以增强模型对当前题目的理解；[Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)
- 我们使用了Duxiaoman开源的[FinCorpus](https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus/tree/main/data)作为大规模的金融试题训练集，并对题目进行清洗，包括去重、过滤低质量题目、文本结构重构等等，最后得到约20万道金融考试题目；
- 我们采用[bge-embedding](https://huggingface.co/BAAI/bge-large-zh-v1.5)语义编码每一道金融例题并得到Embedding，保存到向量数据库中；
- 我们针对FinKBenchmark里每一道CPA单选题和CPA多选题进行编码并在数据库进行向量相似检索，得到5道相似的例题作为少样本示例插入到Prompt中；
- 如下是一个使用了dynamic few-shot的样本，通过在上下文中补充 **借款** 或 **费用** 的相关知识，帮助模型完成当前题目。

```
当前题目：
下列各项中，不属于借款费用的是（　）。
A. 企业发行债券产生的折价或者溢价的摊销额
B. 以咨询费名义向银行支付的借款利息
C. 外币借款汇兑差额
D. 发行股票支付的承销商佣金、手续费
答案
```

- Dynamic few-shot补充到相似题目到上下文：

```
prompt:
以下是关于accounting考试的单项选择题，请选出其中的一个正确答案。

下列项目中，不属于借款费用的是（ ）。
A、外币借款发生的汇兑差额
B、应付债券计提的利息
C、发行债券所发生的溢价
D、应付债券折价的摊销
答案：C

下列各项中，不属于“财务费用”科目核算内容的是（ ）。
A、短期借款利息支出
B、销售商品发生的现金折扣
C、办理银行承兑汇票支付的手续费
D、发生的业务招待费
答案：D

[单选题]下列各项中,属于资金占用费的是()。
A. 借款手续费
B. 债券利息费
C. 借款公证费
D. 债券发行费
答案：B

下列费用中，不属于期间费用的是（ ）。
A、管理费用
B、销售费用
C、待摊费用
D、财务费用
答案：C

下列各项中，不属于借款费用的是（　）。
A. 企业发行债券产生的折价或者溢价的摊销额
B. 以咨询费名义向银行支付的借款利息
C. 外币借款汇兑差额
D. 发行股票支付的承销商佣金、手续费
答案：
response:      D
ans:           D
ground truth:  D 
```

- [Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)针对动态的少样本示例还插入了推理步骤，也就是例题的解析，辅助GPT-4采用思维链对当前题目进行推理；
- 根据实验总结，FinKBenchmark上大部分14B规模以下的开源模型启动思维链会适得其反，与[FinEval](https://arxiv.org/abs/2308.09975)的技术报告观察一致；
- 因此我们的cpa_one_rag和cpa_multi_rag数据集 **暂不考虑** 加入例题的解析作为少样本思维链推理的示例。
