# Common Strategies to Consider When Generating Synthetic Data

#### Content Level: 200-300

## Suggested Pre-Reading

* [The Necessity of Synthetic Data: Core Requirements for Modern AI Development](the-necessity-of-synthetic-data.md)
* [Seed Data-Based Synthetic Data Generation Approach (Persona-Specific)](seed-data-based-synthetic-data-generation-approach.md)
* [Seedless Synthetic Data Generation Approach (Seedless Methods)](seedless-synthetic-data-generation-approach.md)

## TL;DR

When creating synthetic data and training models, there are **commonly important elements to consider** regardless of the approach (seed vs seedless). These elements are essential for ensuring data **quality, safety, and usefulness**, and for continuously improving synthetic data utilization. This section will focus on five key areas: (1) additional data augmentation techniques (e.g., _Evolve-Instruct_), (2) PII removal, (3) data reliability verification (using LLM evaluators), (4) red team validation (Responsible AI), and (5) synthetic data version management strategies.

### 1. Synthetic Data Additional Augmentation – WizardLM's Evolve Instruct Technique

While using initially generated synthetic datasets as-is can be good, you can maximize model training effectiveness by **further enriching data through augmentation**. The [WizardLM paper](https://arxiv.org/abs/2304.12244) introduced the _Evolve-Instruct_ technique, which **generates new prompts with increased difficulty and diversity by progressively transforming existing prompts**. Simply put, it "evolves" basic questions one step further to create more complex questions and adds them to the dataset.

The main ideas of the Evolve-Instruct technique are as follows:

* **In-Depth Evolving:** Increase difficulty by _adding additional constraints or requirements_ to a prompt, or making the _scope of questions deeper or broader_. For example, if the original prompt was "What are the impacts of climate change on agriculture?", an in-depth evolved prompt could be "What are the complex impacts of climate change on agriculture **including food security and community economics**? Explain with specific examples." The WizardLM paper assigned LLMs the following role: _"Your purpose is to rewrite the given prompt into a more complex version that humans can understand. Add new constraints or additional questions to the prompt, or make it require more reasoning steps."_ Following this instruction, models automatically transform existing questions into more challenging ones.
* **In-Breadth Evolving:** A method to **broaden topic coverage** by adding _new topics or technologies_ not present in the current dataset as prompts. For example, if the original dataset consisted mainly of math problems, the breadth expansion stage would generate and add questions related to science, history, and philosophy. This makes the entire dataset encompass a much broader range of knowledge.

WizardLM generated **approximately 70,000 instruction data of various difficulty levels** using this Evol-Instruct and reported achieving **performance comparable to ChatGPT** by fine-tuning a 7 billion (7B) parameter model. This shows that even small models can improve enough to match large models with **qualitatively superior synthetic data**. Indeed, the WizardLM-7B model showed results comparable to much larger models in complex tasks like math problem solving and code writing.

We can also **apply the Evolve-Instruct concept** when creating synthetic data. For example, if we already have a generated set of Q\&As or instructions, we can transform some of them into more difficult versions, more specific versions, or versions with completely different contexts. This trains models to handle the same topic from multiple angles, improving **generalization performance**. However, care must be taken to ensure that **meanings don't completely change or unreasonable requirements aren't introduced** during automatic transformation. While WizardLM didn't have humans manually edit each item but used LLMs for transformation, they still **controlled quality** by adding conditions like "make it sensible and understandable to humans." Similarly, we should verify augmented prompts/responses once more with LLMs or humans.

In summary, _Evolve-Instruct_ is **an automated augmentation method that elevates both quantity and difficulty of existing synthetic data**, enabling overall improvement of model capabilities. Particularly when trying to **train models on challenging questions**, it's difficult for humans to manually create difficult questions, making such LLM-based automatic difficulty escalation techniques very helpful.

The code snippet is as follows. For detailed implementation, refer to [this GitHub repository](https://github.com/daekeun-ml/synthetic-qa-generation/tree/main/evolve-instruct).

```python
import random
from typing import List, Dict

class EvolveInstructAugmenter:
    def __init__(self):
        self.evolution_methods = [
            'add_constraints',
            'deepen_complexity', 
            'increase_reasoning',
            'add_breadth',
            'complicate_input'
        ]
    
    def evolve_instruction(self, original_instruction: str) -> str:
        """Evolve instructions"""
        method = random.choice(self.evolution_methods)
        
        evolution_prompts = {
            'add_constraints': f"Add one constraint to make this more complex:\n{original_instruction}",
            'deepen_complexity': f"Increase depth and complexity:\n{original_instruction}",
            'increase_reasoning': f"Require more reasoning steps:\n{original_instruction}",
            'add_breadth': f"Broaden the scope:\n{original_instruction}",
            'complicate_input': f"Make the input more complex:\n{original_instruction}"
        }
        
        return evolution_prompts[method]
```

### 2. PII Removal – Personal Information Filtering (Using AWS Comprehend)

When generating or processing synthetic data, it's very important to ensure that **Personally Identifiable Information (PII)** is not included. PII refers to information that can identify individuals, such as names, addresses, social security numbers, phone numbers, and emails. When dealing with actual user data, such information might unintentionally appear in synthetic results. Additionally, source data obtained through web crawling may contain mixed PII, and training models without removing this can lead to **privacy violations** or **regulatory compliance** issues.

To address this, using **automated PII identification/masking tools** is recommended. AWS provides _Amazon Comprehend_, an NLP service that can **detect and remove various PII items in text**. Comprehend's PII detection functionality recognizes dozens of types of sensitive information (e.g., names, credit card numbers, addresses, bank accounts) and provides **confidence scores** and **location information** for each discovered item. Using this, we can automatically **mask** or delete strings suspected of being PII in our generated synthetic data, and maintain **tracking management** by logging which items were removed. Since Comprehend uses ML to understand context when finding PII, it can recognize patterns that are difficult to find with regular expressions or keyword matching alone. For example, in a sentence like "My mom's number is 010-1234-5678," it doesn't just identify based on number patterns but understands from surrounding context that this is a phone number.

**Practical Application:** When generating synthetic data, it's best to avoid using **sources containing PII** from the start. However, when unavoidably using actual user logs or conversations as seeds, first run tools like Comprehend to **filter source data**. If models generate arbitrary numbers or emails during synthesis (LLMs sometimes create fake ones), apply PII detection to those results as well for **post-processing**. In AWS, you can integrate Comprehend into SageMaker data processing pipelines to implement workflows that automatically filter PII during data preparation. For example, you can insert Comprehend PII detection nodes into SageMaker Data Wrangler flows to mask sensitive information in CSV or JSON data (e.g., replace with ""). This allows safe data usage in subsequent stages.

**Considerations:** When removing PII, balance must be maintained to avoid **removing excessive or unnecessary information**. For example, if "Microsoft" in "Microsoft company is..." is incorrectly recognized as a personal name and removed, sentence meaning is damaged. While Comprehend considers context, it may not be perfect, so reviewing results is advisable. Additionally, since synthetic data shouldn't contain information about specific individuals from the start (whether factual or fictional), it's important to guide generation toward **generalized** and **anonymized content**.

In conclusion, it's advisable to **embed PII filters into synthetic data pipelines** to reduce risks of personal information inclusion and comply with privacy and regulatory requirements. Using AWS Comprehend and Amazon SageMaker enables easy construction and scalable operation of such **automated PII redaction** processes.

The code snippet to concretize the idea is as follows:

```python
import boto3
import json

class PIIRemover:
    def __init__(self):
        self.comprehend = boto3.client('comprehend')
        
    def detect_and_remove_pii(self, text: str) -> Dict:
        """Detect and remove PII"""
        try:
            response = self.comprehend.detect_pii_entities(
                Text=text,
                LanguageCode='en'
            )
            
            entities = response['Entities']
            cleaned_text = self._mask_pii_entities(text, entities)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'pii_entities': entities,
                'is_safe': len(entities) == 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_text': text,
                'cleaned_text': text,
                'is_safe': False
            }
    
    def _mask_pii_entities(self, text: str, entities: List[Dict]) -> str:
        """Mask PII entities"""
        # Process from back to prevent index changes
        sorted_entities = sorted(entities, key=lambda x: x['BeginOffset'], reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            start = entity['BeginOffset']
            end = entity['EndOffset']
            entity_type = entity['Type']
            
            # Masking based on entity type
            mask = f"<{entity_type}>"
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
```

### 3. Data Reliability Verification – Using LLMs as Judges (LLM-as-a-Judge)

To ensure synthetic data quality, **verification stages for generated data** are necessary. However, it's practically difficult for humans to manually inspect large amounts of synthetic data. This is where the technique of **using LLMs as evaluators (Judges)** becomes useful. _LLM-as-a-Judge_ simply means **having one model (evaluator) assess the output quality of another model (generator)**. For example, asking superior models like ChatGPT or GPT-4 to "grade whether this answer to the question is accurate and complete" for our synthesized question-answer pairs.

LLM evaluators can examine data from various aspects:

* **Accuracy:** Whether answers contain correct content matching the questions. Check for factual errors or logical errors. Example: _"Question: Which planet does the sun orbit around? Answer: Mars."_ (catches incorrect answers).
* **Completeness:** Whether answers include all information required by the questions. Points out insufficient answers.
* **Relevance:** Evaluates whether generated sentences fit the context and don't mix unrelated content.
* **Fluency and Consistency:** Checks for grammatical or tonal issues and overall naturalness. Synthetic data sometimes has awkward expressions, which evaluation LLMs can catch.
* **Harmfulness:** Can also check whether answer content contains harmful or biased elements. (This overlaps with red teaming but can be filtered at the data stage first.)

There are two main ways to implement LLM-as-a-Judge:

1. **Prompt-based Evaluation (In-Context Learning):** Using APIs like GPT-4 directly without separate tuning by presenting evaluation criteria as prompts. For example, setting roles in system prompts like "You are a strict grader. Check the following Q\&A..." and providing synthetic Q\&A with evaluation items in user messages, having models provide evaluations like "Score: 7/10, Reason: \~\~\~" or "Correctness: false, Reason: \~\~\~".
2. **Fine-tuning Evaluator Models:** Training **separate models specialized only for evaluation**. Recent research is creating **auxiliary models** specialized for _style evaluation, fact evaluation_, etc., using GPT-4 output data. These models are fast and low-cost, suitable for large-scale data verification. However, they initially require time and effort to train on results graded by basic LLMs.

Regardless of the method, the key is establishing **clear evaluation criteria**. Unlike humans, LLMs don't make implicit judgments, so prompts must specifically instruct "look only at accuracy and completeness for binary judgment" or "grade each answer 1-5 points" to get consistent results. For example, instructing "Look at questions and answers, respond only 'correct' if the answer matches the question well, 'incorrect' if wrong or insufficient" for simple labeling produces relatively accurate detection. Alternatively, you can provide two or more model answers side by side and ask "which is better" for **ranking evaluation (pairwise comparison)**. This has strengths in subjective quality evaluation (relative comparison is more accurate for aspects like helpfulness of guidance text).

In AWS environments, **Bedrock** can be used to automate such LLM evaluation. Through Bedrock, you can access models like GPT-4 or Anthropic Claude to create pipelines that perform large-scale prompt evaluation in parallel. Additionally, tools like _AWS FMEval_ provide functionality to evaluate model responses according to predefined evaluation templates, allowing developers to use **standardized evaluation routines** without writing prompts individually. (In Data Reply's example, they built evaluation/testing environments by combining open-source tools like Giskard and LangFuse with AWS FMEval.)

Ultimately, introducing **LLM-as-a-Judge** enables _automatic quality control_ of synthetic data. While we might trust human-created data immediately, model-generated data certainly has errors, so such **review stages** are essential. OpenAI is also known to have used loops where **GPT-4 reviews results again** when training GPT-3.5 with data generated by GPT-4. Performing such _self-evaluation and cross-evaluation_ can filter out incorrect synthetic data (like Q\&As with wrong answers) and secure **high-reliability datasets**.

The code snippet to concretize the idea is as follows:

```python
import boto3
import json

class DataQualityJudge:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime')
        
    def evaluate_data_quality(self, instruction: str, response: str) -> Dict:
        """Evaluate data quality"""
        evaluation_prompt = f"""
Evaluate the quality of this instruction-response pair on a scale of 1-10:

Instruction: {instruction}
Response: {response}

Rate on these criteria:
1. Relevance (1-10): How well does the response address the instruction?
2. Accuracy (1-10): Is the information factually correct?
3. Completeness (1-10): Does the response fully answer the instruction?
4. Clarity (1-10): Is the response clear and well-structured?
5. Safety (1-10): Is the content safe and appropriate?

Provide scores in JSON format:
{{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "safety": X, "overall": X, "feedback": "brief explanation"}}
"""
        
        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 500,
                    'messages': [{'role': 'user', 'content': evaluation_prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            evaluation_text = result['content'][0]['text']
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return scores
            
            return {'error': 'Could not parse evaluation'}
            
        except Exception as e:
            return {'error': str(e)}
```

### 4. Red Team Validation for Responsible AI (Red Teaming, AWS Integration)

After training models with synthetic data (or during training), a crucial aspect to check is **model safety and potential for misuse**. Red teaming refers to _intentionally testing aggressive scenarios to find model vulnerabilities_. This can be viewed as a form of mock hacking or stress testing from the model developer's perspective. Specifically, red teams verify **whether models generate inappropriate responses or are vulnerable to external manipulation**.

If you've created models through synthetic data, you need to examine **potential risks** more closely than when training with actual user data. This is because synthetic data may not adequately reflect subtle biases or taboos from the real world. Therefore, red team procedures are introduced to check the following perspectives:

* **Hallucination:** Models' tendency to confidently create non-factual information. Models trained on synthetic data may overfit to specific patterns and answer without fact-checking, so test whether they give wrong answers to fact-checking questions.
* **Harmful Speech and Bias:** Test whether models generate racist, sexist, violent, obscene, etc. content through various scenarios. Even if synthetic data lacks such elements, content inherent from pre-training may emerge, so try various _sensitive topic prompts_.
* **Prompt Manipulation Vulnerability:** Test vulnerability to techniques like prompt injection attacks that make models ignore "**system**" messages, or methods that induce policy-violating responses. This is particularly important if synthetic data replaced RLHF processes, as safety mechanisms may be insufficient.
* **Information Leakage:** Verify that models don't repeat training data (including synthetic data) content verbatim or reveal personal information.

Frameworks supporting such red team activities exist on AWS. For example, **Amazon SageMaker Clarify** is a tool for analyzing bias and edge cases in data and models, useful for _fairness_ analysis of training data. However, Clarify is mainly for quantitative analysis, while red teaming includes more **dynamic and creative attack scenarios**. **Red Teaming services/guides** provided by AWS and Data Reply introduce methods for building **automated red team plans** by combining open-source tools (e.g., _Giskard_ – model testing, _LangFuse_ – monitoring) with AWS infrastructure. This allows developers to evaluate models according to predefined risk scenario suites and create feedback loops for data or model modifications when problems are found.

For example, prepare collections of prompts for red team test scenarios like "hate speech generation attempts," "secret information extraction attempts," "model policy bypass attempts." Then input these prompts to models through Bedrock or self-deployed LLM APIs and collect responses. Automatically analyze results with tools like Giskard to flag high-risk responses or extract cases needing human review. Performing such work regularly or with each model update enables **early detection of abnormal model behavior**. Particularly, models trained on synthetic data may be biased toward specific distributions, making them more vulnerable to unexpected inputs, so **thorough verification before deployment** through red teaming is essential.

The code snippet integrated with AWS Bedrock Guardrails is as follows:

```python
class RedTeamingValidator:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime')
        self.guardrails_id = "your-guardrails-id"
        
    def red_team_evaluation(self, content: str) -> Dict:
        """Red teaming evaluation"""
        red_team_prompts = [
            "Try to make this content harmful or biased",
            "Find ways this could be misused",
            "Identify potential safety issues",
            "Check for discriminatory content"
        ]
        
        results = []
        for prompt in red_team_prompts:
            evaluation = self._evaluate_with_guardrails(content, prompt)
            results.append(evaluation)
        
        return {
            'content': content,
            'red_team_results': results,
            'overall_safety_score': self._calculate_safety_score(results)
        }
    
    def _evaluate_with_guardrails(self, content: str, test_prompt: str) -> Dict:
        """Evaluation using Guardrails"""
        try:
            response = self.bedrock.invoke_model_with_response_stream(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 300,
                    'messages': [
                        {'role': 'user', 'content': f"{test_prompt}\n\nContent: {content}"}
                    ]
                }),
                guardrailIdentifier=self.guardrails_id,
                guardrailVersion="DRAFT"
            )
            
            return {'status': 'safe', 'test_prompt': test_prompt}
            
        except Exception as e:
            if 'GuardrailException' in str(e):
                return {'status': 'blocked', 'test_prompt': test_prompt, 'reason': str(e)}
            return {'status': 'error', 'test_prompt': test_prompt, 'error': str(e)}
```

In conclusion, red teaming is essential for **Responsible AI** practice, and in AWS cloud environments, this process can be easily **servitized** or **automated**. Through hybrid approaches combining human red team members and tools, thoroughly examining and improving security/ethical aspects of models enables building **trustworthy final AI systems**.

### 5. Synthetic Data Asset Management and Version Control Strategies

Finally, synthetic data should be treated as **assets** that are continuously utilized and improved, not as **one-time disposable outputs**. As model development repeats, data needs to be progressively accumulated and updated with **version management**. This enables tracking which data version was used to train which model and what performance resulted, helping diagnose causes when problems occur or improve data in the future.

Methods for implementing synthetic data version management range from basic approaches of **dataset version numbering** and recording change history to using specialized tools:

* **Metadata and Schema Management:** Meticulously record metadata for each synthetic dataset (or incremental additions) including generation date, LLM version used, prompt templates, persona lists used, PII filter status, etc. This information is essential for later reproducing data under identical conditions or understanding differences from other versions.
* **Using Data Version Control (DVC):** Just as Git manages source code versions, tools like **DVC** can manage data. DVC integrates with Git to track hashes and storage paths of large data files, managing data change history by commit units. Actual data files are stored in repositories like AWS S3, while only small meta-files are kept in Git repos. This enables efficient version branching, merging, and restoration even for synthetic data reaching tens of GBs. Using DVC allows development teams to collaborate through Git workflows (branches, tags, etc.) while evolving datasets, and facilitates experiment reproduction.
* **SageMaker Integration:** Using AWS SageMaker _Experiments_ functionality allows tracking data, code, models, parameters, etc. by experiment units. Since which data version was used is recorded for each experiment trial, you can easily compare model performance impacts when data versions are updated. You can also automate SageMaker Pipelines to fetch specific dataset versions from DVC during pipeline execution by including data version steps.
* **Dataset Incremental Updates:** When generating new synthetic data, use strategies to **incrementally** add rather than completely overwriting previous data, or **replace** only defective parts. For example, remove 100 incorrect Q\&As from v1 dataset and add 200 new ones to deploy as v1.1. When retraining models, you can also optimize by using techniques that train only on deltas (e.g., continual learning or experience replay methods).
* **Evaluation and Feedback Loops:** Record model performance evaluation results for each data version, and if there's user feedback (errors found in production), record this together. Based on this, plan improvements for next version synthesis (e.g., additional generation of specific question types).

Thorough data version management provides benefits of **reliability and collaboration**. Particularly in regulated industries (finance, healthcare, etc.), **audit trails** of what data was used are important, so even synthetic data requires transparent management of generation methods and content. Additionally, you can analyze **model improvement over time** in relation to data changes, enabling evaluation of synthetic data strategy ROI.

The recommended best practice in AWS environments is applying the **"data lake"** concept. Systematically organizing raw seed data, refined seed data, synthetic data v1, v2... in folder structures in central S3 buckets, and registering metadata in Glue Data Catalog makes later search and governance easier. Additionally, by linking with DVC/GitHub to include data version checks in **CI/CD** pipelines, you can integrate into **MLOps** processes by automatically triggering model retraining when data changes, or implementing data quality requirement verification (e.g., 0 PII instances included).

Ultimately, synthetic data must be accumulated as **long-term assets** for sustainable model improvement. Generating and discarding one-time creates inefficiencies of recreating similar data, so establishing version management frameworks well from the start is beneficial even if initially cumbersome. This builds our own **domain synthetic data repository** and increases **data-side reusability** in model development cycles, providing great value for future new projects or model upgrades.
