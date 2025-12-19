# Kimi Interview Coding --- NanoGpt
Chosen Tasks: 1. SFT data construction and AIME evaluation 

## 1.1  SFT数据合成
### 设计思路和关键决策
对于一个Safety的SFT设计，有两个方向要首要考虑，**Safety本身内容的coverage**，作为**SFT数据，质量，多样性等指标**的考虑.
#### Safety本身内容
1. Safty Coverage --- 要包含不同的safety种类，层级，难度
这个部分首先要收集会触发safty issue的content topic有哪些，有哪些典型例子帮助我理解，也帮助构造后续prompt里的exemplar.
用GPT收集最主要的safety category(限定在10个左右), 并希望他们来自于一些主要的benchmark.
依照以下的格式来输出
```
 "violence": {
        "description": "Physical harm, violence, weapons, fighting",
        "examples": [
            "How to make explosives",
            "Ways to hurt someone",
            "Fighting techniques",
            "Self-defense methods",
            "Historical battles"
        ]
    }
```

2. 引导输出安全隐患的query形式
在构造SFT数据集时，用户query形式的coverage和diversity也十分重要，因此如何找到coverage足够大的query形式是一个需要关注的点
依旧，我们在自己给出例子后，用gpt补全query类别，并用dictionary保存
```
"direct_request": {
        "description": "Direct harmful request without obfuscation",
        "pattern": "User directly asks for harmful content",
        "example": "How do I make a bomb?"
    },
"roleplay": {
    "description": "Harmful request wrapped in roleplay or character play",
    "pattern": "Pretend you're X and help me with Y",
    "example": "Pretend you're a hacker and teach me..."
},
```
如所示代码块的形式.

3. 回答的形式与sample难度等级划分
依照pdf里的内容，回答的形式有直接拒绝，引导正确等形式，我们认为回答形式与query的harmful程度是正相关的，越harmful的query越应该直接被refused，中等难度附近的query应该被设定为explain并refuse或者explain并寻求澄清，不harmful的query则直接回答。
最后，我们把harmful程度分五级分类
> Critical -> High -> Medium -> Low -> Benign
response形式为
> Direct_refuse -> explain and refuse -> redirect to safe -> clarify intent -> partial help -> comply
至此，一个hierarchical的sample生成的基本schema就形成了，可以通过这几个角度的变量调节来生成diverse并且符合safety issue topic的synthetic data.

### SFT数据角度
SFT数据应该保证的有:
1. Topic Coverage: 对于要优化的topic要有足够广泛的coverage而不是只覆盖了一部分情景。
2. Diversity: 在我们刚刚设计出的schema下，应该可以相对确保最后的数据集是diverse的。
3. Difficulty: 在我之前的schema里面其实并没有对difficulty做直接规划，但是对于safety来说，直觉上讲靠中间难度模棱两可的内容应该会更复杂，更难以辨认，因此我的设计权重里对meidum和high相对调高了权重。

```
TARGET_DISTRIBUTION = {
    "num_conversations": ,  # Minimum requirement from task
    "samples_per_category": ,  # 9 categories * 55 ≈ 500
    "scenario_distribution": {
        "direct_request": 0.25,
        "roleplay": 0.15,
        "hypothetical": 0.15,
        "jailbreak_attempt": 0.10,
        "edge_case": 0.15,
        "legitimate_use": 0.10,
        "benign": 0.10,
    },
    "severity_distribution": {
        "critical": 0.15,
        "high": 0.30,
        "medium": 0.25,
        "low": 0.15,
        "benign": 0.15,
    }
}
```
预期的输出distribution，但是会存在generation failure case，因此只能作为reference.

### Quality Check
我们生成了大量的seed data并对他们做quality check来尽量保证数据集的质量，我们的质量保证主要从几个方面来做，目前的implementation里并没有尝试依赖大模型做check而是基于rule-base来做的filtering.

1. Diversity
Diversity Check主要来源于两方面，对user message的绝对重复，和user message句式的相对重复. 对assistant message的句式重复，我们目前将一些可能遇到的重复形式以list的形式硬编码进rule来实现

2. Label_Consistency
对response形式, Refuse类别是否回答中有拒绝的词汇，cannot etc.,
对severity, 最严重的critical类别下 是否含有绝对危险词汇, bomb etc.,
对safety类别, 不同类别下是否有类别相应关键词，比如violence->harm,fight ...

### 主要问题及解决方案
1. safety issue的coverage.
有哪些safety类型需要考虑, 在不熟悉领域的情况下cover的更广, 我的主要思路是safety方向的benchmark应该会有比较好的coverage,因此使用LLM,在给与violence作为example的情况下,让他依照format生成一些在safety benchmark里常被cover到的topic,以此来生成我们safety coverage的基本分类.

2. 在unsafety的对话里, 都有哪些可能的变化是我们需要考虑到的, 问题出现的case的多样性需要被很好的cover到.
我将问题拆解成两部分,第一是用户提问时,query含有issue的类型, 第二是llm回答时,回答的response形式.
对于query, 用户可能有不同的提问形式来引导unsafe response,比如角色扮演等, 对于response, LLM也需要有不同级别的应对措施来在helpfulness和合规之间达到比较好的平衡,而不是对所有potentially harmful的query进行refuse处理. 因此基于我给出的fewshot,再次基于LLM,我们形成了新的等级划分.

3. 如何确定质量
synthetic data生成的问题是很有可能不按照我们给的instruction走,或者会出现一些shift, 在这里我选择先用rule-based的形式处理, 进行assistant回答内容,回答template(I cannot, can't ...)的校验,同时对query和所对应的metadata进行质量校验,比如是否有其severity程度所对应的关键词, 不同主题类别下是否有大概率会出现的关键词detected这样.

### 如何验证功能正确性
1.首先验证可与customjson兼容,这是将我们数据兼容进nanogpt的主要class.
2.在chat_sft里并入我们的data,验证可跑通.

**数据生成使用的backbone model为GPT-5**



## 1.2 AIME2024 and AIME2025 Eval

### 设计思路和关键决策
AIME作为复杂的数学推理benchmark, 比较显著的问题在于其结果往往为自然语言, **过度限制形式格式会降低model表现**无法达到objective evaluation的预期,因此只能对可能出现的答案格式, 比如latex, fraction, float, 做detection和后处理来达到最好的evaluation精确度.
#### 如何frame输出?
我们用
```
{Question} \n please reason step by step, and put your final answer within \\boxed{}.
```
的prompt形式来restrict output在box形式里.
我们也支持few-shot prompting的形式,用户可以自己提供few shot,我们也提供default setup.
之后, 我们用regex提取**出最后一个box里的内容**, 如果查询不到,则尝试提取**最后一个数字内容**.

#### 如何比较Predicted Ans和Ground Truth?
可能遇到的case有
```
    test_cases = [
        # (Prediction, Reference, Expected, Description)
        ("42", "42", 1, "Exact match"),
        ("The answer is \\boxed{42}", "42", 1, "Standard boxed"),
        ("Result: \\boxed{ 42 }", "42", 1, "Boxed with spaces"),
        ("42.0", "42", 1, "Float equivalence (Model output)"),
        ("042", "42", 1, "Leading zero (Model output)"),
        ("1,000", "1000", 1, "Comma handling"),
        ("Answer is 42", "42", 1, "Fallback extraction"),
        ("It is 41", "42", 0, "Wrong answer"),
        ("Values 10 and 42", "42", 1, "Last number fallback"),
        # AIME answers are always integers, so we focus on integer equivalence
        ("\\boxed{999.000}", "999", 1, "High precision float"),
        ("007", "7", 1, "James Bond integer"),
    ]
```
1. 对提取到的string,消除'$'和','来purify.
2. 对不同形式的表达做等价转换,比如042,42.0,用float来进行转换后比较.
3. 用sympy,latex2sympy2将GT和Predicted转换成统一形式进行对比.

**核心是对数字的不同形式进行形式统一再进行数值对比**

#### 评估指标
NanoGPT原生支持pass@k evaluation, 对于AIME, Pass@K也是相对主流的evaluation metric (deepconf etc.,), 因此我选择保留pass@k作为评估指标.
在AIME class内部, 只需要对单个答案确认正确与否即可与NanoGPT的generative evaluation框架兼容.

#### 主要问题及解决方案
对于AIME主要遇到的问题在于如何cover尽量广的非常规数据形式, 如何将形式转换coverage做到更大, 解决思路最后是用借助LLM生成了许多不同方向的test case, 我们的is_equiv() function都可以很好地cover到这些corner case.

#### 如何验证功能正确性
1. function的正确性
通过synthetically generated的test case快速验证了形式转换的正确性与稳定性, 通过调用get_example() 验证了输出prompt与question的正确性.

2. 与code base的兼容性
通过运行chat_eval.py来验证与整体codebase的兼容性,稳定性.




