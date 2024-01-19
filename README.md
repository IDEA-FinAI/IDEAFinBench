# FinKnowledgeEval: 金融知识评估基准

# ✨ 简介

近年来，通用大模型（LLMs）在人工智能领域取得了显著进展，尤其是在处理复杂的自然语言任务方面。然而，金融领域对这些模型的具体应用和测试仍然相对缺乏。为了补充这一空白，我们提出了一种新的评估基准：**FinKnowledgeEval**。

FinKnowledgeEval旨在评估通用大模型在处理金融相关知识和问题时的能力。此基准专注于金融专业知识，包括但不限于会计、审计、投资、经济学、税法等方面。FinKnowledgeEval包括两个主要测试集，分别针对CPA（注册会计师）和CFA（特许金融分析师）考试，旨在考察通用大模型在金融领域的知识储备水平和逻辑推理能力。

通过FinKnowledgeEval，我们希望能够更深入地了解通用大模型在处理复杂和专业化金融知识方面的能力。这不仅有助于推动金融领域的人工智能应用发展，也为未来更广泛的行业应用提供了重要的基础。

# ❤️ 数据集介绍

## **CPA-Eval数据集**
  
  CPA-Eval数据集涉及会计、财务成本管理、税法、审计、公司战略与风险管理和经济法六大科目，全面考察通用大模型在会计、审计、税法等领域的能力。

  从题目特点来看，六大科目可以分为 **计算型** （评估模型的金融逻辑推理能力）和 **记忆型** （评估模型的金融知识储备水平）。其中计算型主要包括会计、财务成本管理和税法，记忆型主要包括审计、公司战略与风险管理和经济法。

  从题目类型来看，测试题目可以分为 **单项选择题** 和 **多项选择题** 。相比于单项选择题，多项选择题要求通用大模型从多个选项中选择一个以上的正确答案，更加考验通用大模型的综合分析和判断能力。同时，多项选择题的评估方式也更加复杂。
  
![CPA-Eval](assets/CPA-Eval.jpg)

下面是 会计科目 **单项选择题** 和 **多项选择题** 示例：

- **单项选择题** 

```
id: 0
question: 下列各事项中，各公司应按照股份支付会计准则处理的是（ ）。
A: 大海公司以自身普通股授予其子公司管理人员
B: 飞鸟公司分配现金股利给其股东
C: 青山公司租赁房屋给在职员工免费使用
D: 绿水公司用外购产品分配给在职员工
answer: A
expanation:
解析：股份支付，是指企业为获取职工和其他方提供服务而授予权益工具或者承担以权益工具为基础确定的负债的交易。选项A：大海公司以自身普通股授予其子公司管理人员，属于以权益结算的股份支付，应按照股份支付会计准则处理，因此，选项A正确。选项B：飞鸟公司分配现金股利给其股东，属于对已有权益的股东进行分红，不适用股份支付准则。因此，选项B错误。选项CD：租赁房屋给在职员工免费使用、用外购产品分配给在职员工均属于非货币性福利，应按照职工薪酬准则进行会计处理，因此，选项CD错误。综上所述，本题答案为选项A。
```

- **多项选择题** 

```
id: 0
question: 企业使用信用衍生工具管理金融工具（或其组成部分）的信用风险敞口时，将其指定为以公允价值计量且其变动计入当期损益的金融工具需要满足的条件有（ ）。
A: 金融工具信用风险敞口的主体（如借款人或贷款承诺持有人）与信用衍生工具涉及的主体相一致
B: 金融工具的偿付级次与根据信用衍生工具条款须交付的工具的偿付级次相一致
C: 若不进行指定，将会产生会计错配
D: 衍生工具的信用风险无法单独识别
answer: A,B
expanation:
解析：企业使用以公允价值计量且其变动计入当期损益的信用衍生工具管理金融工具（或其组成部分）的信用风险敞口时，可以在该金融工具（或其组成部分）初始确认时、后续计量中或尚未确认时，将其指定为以公允价值计量且其变动计入当期损益的金融工具，并同时作出书面记录，但应当同时满足下列条件：
（1）金融工具信用风险敞口的主体（如借款人或贷款承诺持有人）与信用衍生工具涉及的主体相一致（选项A）；
（2）金融工具的偿付级次与根据信用衍生工具条款须交付的工具的偿付级次相一致（选项B）。
因此，选项AB正确；选项CD错误。
综上所述，本题答案为选项AB。
```

## **CFA-Eval数据**
  
  CFA-Eval数据集包含 **Level 1** 和 **Leval 2** 两个层次的测试数据，涉及道德与专业准则、量化方法、经济学、财务报表分析、公司金融、权益投资、固定收益、衍生品、另类投资、投资组合管理等十大科目，全面考察通用大模型对于经济、金融、资产管理等方面知识的理解能力，同时考察大模型对于真实金融案例的分析能力。

  从考试级别来看，CFA Level 1 题目以单项选择题为主，一般不涉及复杂的图表，题目相对简单，侧重考察通用大模型对金融基础知识的掌握程度，这是构建高级金融理解的基础；CFA Level 2 题目以案例分析题为主，一般会给出详细的案例背景，并提供与之相关的图表数据，然后针对案例内容考察多道选择题，题目相对复杂，更加强调通用模型在实际应用中的分析、判断和决策能力，特别是在处理复杂的情况和多变量的情景下。

![CFA-Eval](assets/CFA-Eval.jpg)

下面是 另类投资科目 **Level 1** 和 **Level 2** 示例：

- **Level 1** 

```
id: 0
question: Fill in the blanks with the correct words: An American waterfall distributes performance fees on a(n) ___________ basis and is more advantageous to the ___________.
A: deal-by-deal; LPs
B: aggregate fund; LPs
C: deal-by-deal; GP
D: 
answer: C
expanation:
C is correct. American waterfalls, also known as deal-by-deal waterfalls, pay performance fees after every deal is completed and are more advantageous to the GP because they get paid sooner (compared with European, or whole-of-fund, waterfalls). ：C是正确的。美国瀑布，也称为逐笔交易瀑布，在每笔交易完成后支付绩效费用，对GP更有利，因为他们更快获得支付（与欧洲或整个基金瀑布相比）。
```

- **Level 2** 

```
id: 0
question:
Hui Lin, CFA is an investment manager looking to diversify his portfolio by adding equity real estate investments. Lin and his investment analyst, Maria Nowak, are discussing whether they should invest in publicly traded real estate investment trusts (REITs) or public real estate operating companies (REOCs). Nowak expresses a strong preference for investing in public REITs in taxable accounts.Lin schedules a meeting to discuss this matter, and for the meeting, Lin asks Nowak to gather data on three specific REITs and come prepared to explain her preference for public REITs over public REOCs. At the meeting, Lin asks Nowak:“Why do you prefer to invest in public REITs over public REOCs for taxable accounts?” Nowak provides Lin with an explanation for her preference of public REITs and provides Lin with data on the three REITs shown in Exhibits 1 and 2.The meeting concludes with Lin directing Nowak to identify the key investment characteristics along with the principal risks of each REIT and to investigate the valuation of the three REITs. Specifically, Lin asks Nowak to value each REIT using four different methodologies:Method 1Net asset valueMethod 2Discounted cash flow valuation using a two-step dividend modelMethod 3Relative valuation using property subsector average P/FFO multipleMethod 4Relative valuation using property subsector average P/AFFO multiple
| Exhibit l.Select RE IT Financial Information |
| RE IT A | RE IT B | RE ITC |
| Health |
| Property subsector | Office | Storage | Care |
| Estimated 12 months cash net operating income | $350，000 | $267，000 | $425，000 |
| (NO I) |
| Funds from operations(FFO) | $316，965 | $290，612 | $368，007 |
| Cash and equivalents | $308，700 | $230，850 | $341，000 |
| Accounts receivable | $205，800 | $282，150 | $279，000 |
| Debt and other liabilities | $2，014，000 | $2，013，500 | $2，010，000 |
| Non-cash rents | $25，991 | $24，702 | $29，808 |
| Rec un ng maintenance-type capital expenditures | $63，769 | $60，852 | $80，961 |
| Shares outstanding | 56，100 | 67，900 | 72，300 |

| Exhibit 2.RE IT Dividend Forecasts and Average Price Multiples |
| RE IT A | RE IT B | RE ITC |
| Expected annual dividend next year | $3.80 | $2.25 | $4.00 |
| Dividend growth rate in years 2 and 3 | 4.0% | 5.0% | 4.5% |
| Dividend growth rate(after year 3 into perpetuity) | 3.5% | 4.5% | 4.0% |
| Assumed cap rate | 7.0% | 6.25% | 6.5% |
| Property subsector average P/FFO multiple | 14.4x | 13.5x | 15.1x |
| Property subsector average P/AFFO multiple | 18.3x | 17.1x | 18.9x |


 
Nowak’s most likely response to Lin’s question is that the type of real estate security she prefers:
A: offers a high degree of operating flexibility.
B: provides dividend income that is exempt from double taxation.
C: has below-average correlations with overall stock market returns.
D: 
answer: B
expanation:
REITs are tax-advantaged entities whereas REOC securities are not typically tax-advantaged entities. More specifically, REITs are typically exempted from the double taxation of income that comes from taxes being due at the corporate level and again when dividends or distributions are made to shareholders in some jurisdictions such at the United States.
```

# 📇 模型列表

| Model                  | Size       | Access  | Base Model        |
| ---------------------- | ---------- | ------- | ----------------- |
|ChatGPT                 | -          | API     | -                 |
|GPT-4                   | -          | API     | -                 |
|LLaMA-2-chat            | 7B 13B     | Weights | LLaMA-2           |
|chinese-alpaca-2        | 7B 13B     | Weights | Llama-2           |
|ChatGLM3-Base           | 6B         | Weights | -                 |
|ChatGML3-6B             | 6B         | Weights | ChatGLM3-6B-Base  |
|Baichuan2               | 7B 13B     | Weights | -                 |
|Baichuan2-Chat          | 7B 13B     | Weights | Baichuan2         |
|Qwen                    | 7B 14B     | Weights | -                 |
|Qwen-7B-Chat            | 14B        | Weights | Qwen-7B           |
|Qwen-14B-Chat           | 14B        | Weights | Qwen-14B          |
|Yi                      | 6B         | Weights | -                 |
|Yi-6B-Chat              | 6B         | Weights | Yi-6B             |
|Tongyi-Finance-14B-Chat | 14B        | Weights | Qwen-14B          |
|DISC-FinLLM             | 13B        | Weights | Baichuan-13B-Chat |

# 🚀 模型性能

## **CPA-Eval**

- **Zero-shot**

| 模型                       | 会计    | 审计    | 经济法   | 财务成本管理 | 公司战略与风险管理 | 税法   | 平均    |
| -------------------------- | ----   | ------- | -------- | ---------- | ---------------- | ------- | ------- |
| ChapGPT                    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| GPT-4                      | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Llama-2-7b-chat            | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Llama-2-13b-chat           | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chinese-alpaca-2-7b        | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chinese-alpaca-2-13b       | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chatglm3-6b-base           | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chatglm3-6b                | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-7B-Base          | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-7B-Chat          | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-13B-Base         | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-13B-Chat         | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-7B                    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-7B-Chat               | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-14B                   | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-14B-Chat              | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Yi-6B                      | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Yi-6B-Chat                 | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Tongyi-Finance-14B-Chat    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| DISC-FinLLM                | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |

- **Five-shot**

| 模型                       | 会计    | 审计    | 经济法   | 财务成本管理 | 公司战略与风险管理 | 税法   | 平均    |
| -------------------------- | ----   | ------- | -------- | ---------- | ---------------- | ------- | ------- |
| ChapGPT                    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| GPT-4                      | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Llama-2-7b-chat            | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Llama-2-13b-chat           | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chinese-alpaca-2-7b        | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chinese-alpaca-2-13b       | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chatglm3-6b-base           | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| chatglm3-6b                | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-7B-Base          | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-7B-Chat          | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-13B-Base         | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Baichuan2-13B-Chat         | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-7B                    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-7B-Chat               | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-14B                   | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Qwen-14B-Chat              | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Yi-6B                      | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Yi-6B-Chat                 | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| Tongyi-Finance-14B-Chat    | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |
| DISC-FinLLM                | 00.00  | 00.00   | 00.00   | 00.00       | 00.00            | 00.00   | 00.00   |

## **CFA-Eval**

- **Zero-shot**

| 模型                       | 量化方法 | 经济学   | 财务报表分析 | 公司金融 | 权益投资 | 固定收益 | 衍生品   | 另类投资 | 投资组合管理 | 道德与专业准则 | 平均    |
| -------------------------- | ------- | -------- | ----------- | ------- | ------- | ------- | -------- | ------- | ----------- | -------------- | ------- |
| ChapGPT                    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| GPT-4                      | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Llama-2-7b-chat            | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Llama-2-13b-chat           | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chinese-alpaca-2-7b        | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chinese-alpaca-2-13b       | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chatglm3-6b-base           | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chatglm3-6b                | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-7B-Base          | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-7B-Chat          | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-13B-Base         | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-13B-Chat         | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-7B                    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-7B-Chat               | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-14B                   | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-14B-Chat              | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Yi-6B                      | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Yi-6B-Chat                 | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Tongyi-Finance-14B-Chat    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| DISC-FinLLM                | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |


- **Five-shot**

| 模型                       | 量化方法 | 经济学   | 财务报表分析 | 公司金融 | 权益投资 | 固定收益 | 衍生品   | 另类投资 | 投资组合管理 | 道德与专业准则 | 平均    |
| -------------------------- | ------- | -------- | ----------- | ------- | ------- | ------- | -------- | ------- | ----------- | -------------- | ------- |
| ChapGPT                    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| GPT-4                      | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Llama-2-7b-chat            | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Llama-2-13b-chat           | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chinese-alpaca-2-7b        | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chinese-alpaca-2-13b       | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chatglm3-6b-base           | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| chatglm3-6b                | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-7B-Base          | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-7B-Chat          | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-13B-Base         | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Baichuan2-13B-Chat         | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-7B                    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-7B-Chat               | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-14B                   | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Qwen-14B-Chat              | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Yi-6B                      | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Yi-6B-Chat                 | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| Tongyi-Finance-14B-Chat    | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |
| DISC-FinLLM                | 00.00   | 00.00    | 00.00       | 00.00   | 00.00   |         | 00.00    | 00.00   | 00.00       | 00.00          | 00.00   |




# 🎈 如何进行模型评估














