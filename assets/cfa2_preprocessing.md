# CFA Level2的数据预处理步骤

## **原始的CFA Level2数据集**

```
{
    "first_cat": "章节练习",
    "second_cat": "基础练习题",
    "third_cat": "Economics",
    "section_id": "1564549876669353984",
    "question_id": "1564561456450637825",
    "parent_id": "1564561456450637824",
    "stem": "Hans Schmidt, CFA, is a portfolio manager with a boutique investment firm that specializes in sovereign credit analysis. Schmidt's supervisor asks him to develop estimates for GDP growth for three countries. Information on the three countries is provided in Exhibit 1.「huixue_img/importSubject/1564561401874354176.png」After gathering additional data on the three countries, Schmidt shares his findings with colleague, Sean O'Leary. After reviewing the data, O'Leary notes the following observations:Observation 1:The stock market of Country A has appreciated considerably over the past several years. Also, the ratio of corporate profits to GDP for Country A has been trending upward over the past several years and is now well above its historical average.Observation 2:The government of Country C is working hard to bridge the gap between its standard of living and that of developed countries. Currently, the rate of potential GDP growth in Country C is high.Schmidt knows that a large part of the analysis of sovereign credit is to develop a thorough understanding of what the potential GDP growth rate is for a particular country and the region in which the country is located. Schmidt is also doing research on Country D for a client of the firm. Selected economic facts on Country D are provided in Exhibit 2.「huixue_img/importSubject/1564561402092457984.png」Prior to wrapping up his research, Schmidt schedules a final meeting with O'Leary to see if he can provide any other pertinent information. O'Leary makes the following statements to Schmidt:Statement 1:Many countries that have the same population growth rate, savings rate, and production function will have growth rates that converge over time.Statement 2:Convergence between countries can occur more quickly if economies are open and there is free trade and international borrowing and lending; however, there is no permanent increase in the rate of growth in an economy from a more open trade policy.",
    "issue": "Based upon Exhibit 1, the factor that would most likely have the greatest positive impact on the per capita GDP growth of Country A is:",
    "options": "A.free trade.|B.technology.|C.saving and investment.",
    "answer": "B",
    "analysis": "Country A is a developed country with a high level of capital per worker. Technological progress and/or more intensive use of existing technology can help developed countries increase productivity and thereby increase per capita GDP. Most developed countries have reasonably low trade barriers; thus, somewhat freer trade is likely to have only an incremental, and probably transitory, impact on per capita GDP growth. Also, since the country already has a high capital-to-labor ratio, increased saving/investment is unlikely to increase the growth rate substantially unless it embodies improved technology.<br />",
    "stem_with_img": 1,
    "issue_with_img": 0,
    "analysis_with_img": 0
}
```

- 我们观察到，原始的CFA Level2级题以阅读理解长文为主，且绝大部分题目都包含了以图像形式存储的表格数据：

- 格式较为简单能顺利解析的的表格示例为：

<p align="center"> <img src="https://img.huikao8.com/huixue_img/importSubject/1564561401874354176.png" style="width: 85%;" id="title-icon"></p>

- 格式较为复杂但仍能解析的表格示例为：

<p align="center"> <img src="https://img.huikao8.com/huixue_img/importSubject/1564572973363499008.jpeg" style="width: 85%;" id="title-icon"></p>

- 格式非常复杂以至于无法识别为表格的示例为：

<p align="center"> <img src="https://img.huikao8.com/huixue_img/importSubject/1564574152583680000.png" style="width: 85%;" id="title-icon"></p>

## **调用OCR处理**

- 我们调用了阿里云的[视觉智能表格识别API](https://api.aliyun.com/api/ocr/2019-12-30/RecognizeTable)以识别CFA Level2题干中的表格

- 判断当前题干的图片能否正常识别并解析为表格，采用markdown保存在题干对应的位置，例如

```
| Exhibit 2.RE IT Dividend Forecasts and Average Price Multiples |
| RE IT A | RE IT B | RE ITC |
| Expected annual dividend next year | $3.80 | $2.25 | $4.00 |
| Dividend growth rate in years 2 and 3 | 4.0% | 5.0% | 4.5% |
| Dividend growth rate(after year 3 into perpetuity) | 3.5% | 4.5% | 4.0% |
| Assumed cap rate | 7.0% | 6.25% | 6.5% |
| Property subsector average P/FFO multiple | 14.4x | 13.5x | 15.1x |
| Property subsector average P/AFFO multiple | 18.3x | 17.1x | 18.9x |
```
