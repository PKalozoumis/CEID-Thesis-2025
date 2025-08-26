import evaluate
from transformers import AutoTokenizer
from .classes import SummaryUnit

def bert_score(summary_unit: SummaryUnit):

    model = 'roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(model)
    temp = "Childhood obesity is a complex issue, influenced by multiple lifestyle factors and behaviours. Several lifestyle factors are interrelated behavioural contributors to childhood obesity <1923_100-100>. These include diet, physical activity, and stress levels. In fact, the most relevant lifestyle-related risk factors for obesity were selected as intervention targets: (1) diet, (2) physical activity, and (3) stress <1923_5-5>. The consumption of sugar-sweetened beverages is an important contributor to childhood obesity, confirmed by longitudinal studies <1923_62-64>. Additionally, a sedentary lifestyle plays a major role in the rising prevalence of obesity, with lack of physical activity being the fourth leading cause of death worldwide <3611_66-68>. Furthermore, a diet consisting of energy-dense snack consumption, fast-food consumption, and energy-dense drink intake can lead to adverse dietary patterns <1923_79-81>. Childhood obesity is also associated with several immediate health risk factors, such as orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions. Moreover, it has been linked to negative psychosocial outcomes in children, including low self-esteem and depression, which can indirectly impact academic performance and social relationships <272_6-7>. Obesity may also increase the stress level of both the child and their family <1923_90-90>. Interestingly, while obesity was more prevalent among higher socio-economic status (SES) groups, other factors such as family size, residence, and parent's education did not contribute to obesity. However, recent studies have shown a steady increase in prevalence among government school children in large metropolitan cities and peri-urban areas <2581_65-66>."

    summary_unit.summary = temp
    max_len = tokenizer.model_max_length

    print(f"Input tokens: {len(tokenizer(summary_unit.single_text)['input_ids'])}")
    print(f"Summary tokens: {len(tokenizer(summary_unit.summary)['input_ids'])}")
    print(f"Model max length: {max_len}")

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=[summary_unit.summary],
        references=[summary_unit.single_text],
        lang="en",
        model_type=model
    )

    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])

def rouge_score(summary_unit: SummaryUnit):
    pass

    