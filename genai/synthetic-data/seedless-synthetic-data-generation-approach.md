# Seedless Synthetic Data Generation Approach

#### Content Level: 200-300

## Suggested Pre-Reading

* [The Necessity of Synthetic Data: Core Requirements for Modern AI Development](the-necessity-of-synthetic-data.md)
* [Seed Data-Based Synthetic Data Generation Approach (Persona-Specific)](seed-data-based-synthetic-data-generation-approach.md)

## TL;DR

The Seedless approach is a method that creates synthetic data relying solely on LLM knowledge and capabilities **without any pre-prepared data**. It's used when developing AI models for new fields where actual data is scarce, or when training models with very **general knowledge/instruction following** capabilities. In this approach, prompts are structured so that models can produce learning data across a wide range of topics without specific seeds provided by humans.

**Application Scenarios:**

* Developing AI models for completely new domains (when related data is almost non-existent)
* Training general **Instruction-Following** models (developing models capable across various tasks)
* Supporting **low-resource languages** with insufficient data (generating model training data in rare languages)
* **Creative content generation** models (generating novel sentences/ideas without being constrained by existing data)

### 1. Core Ideas of Seedless Synthetic Data Approach

The core of the seedless approach is _"systematically guiding models on what to create"_. Previously, methods would provide a few examples (seeds) and use Self-Instruct techniques to have models create similar data. However, this had limitations where biased seeds would lead to biased results. Recent research trends move toward utilizing **human knowledge structures** to cover the full range of possible tasks.

Particularly, the [**"Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning (GLAN)"**](https://arxiv.org/abs/2402.13064) method proposed in 2024 is an innovative approach that generates **large-scale Instruction tuning data** using only **prior knowledge taxonomy** as input. GLAN creates problems and answers by following a vast **map of human knowledge** without any human-prepared examples or correct answers. Specifically, examining GLAN's procedure:

1. **Knowledge Taxonomy Construction:** First, with help from LLMs like GPT-4, we divide **human knowledge classification systems**. All disciplines and technologies are hierarchized as **field** – **sub-field** – **discipline**. Like creating educational curriculum, we create trees such as "Physics -> Mechanics -> Classical Mechanics" and refine this through human review to create the final **knowledge map**. (This stage has some human intervention, but everything afterward is fully automated by models.)
2. **Subject List Generation:** For each discipline, we ask LLMs (GPT-4) "What subjects should students learn in this field?" We set the model as an expert in that field to output major **subject names, levels, sub-topics** in JSON format. This concretizes the terminal nodes of the vast knowledge system.
3. **Detailed Curriculum (Lesson Plan) Creation:** Now we generate detailed class sessions for each subject and lists of **key concepts** to be covered in those sessions. This also uses LLMs to create "weekly concept lists for teaching subject X." For example, for "Classical Mechanics" subject: Week 1 concepts: Newton's Laws, Week 2: Energy Conservation, etc.
4. **Instruction Generation:** Based on these **detailed concepts**, we generate problems (instructions) that ask about each concept. We prompt LLMs with "Create questions to evaluate whether students understand concept Y" or "Create practice problems including these keywords" to make **task/question sets**. To control difficulty, we sometimes use only one concept per question, or combine two or more concepts for complex problems. (The latter increases difficulty and creates creative application problems.)
5. **Model Response Generation:** Finally, we generate **correct answers/explanations** for the created questions. This part doesn't need the highest-performance models, so the research used GPT-3.5 level for quick responses. While questions might be high-difficulty problems created by GPT-4 level models, answers were optimized to be obtained relatively simply.

Through this process, **numerous Q\&As across virtually all fields** included in the initial knowledge map are generated. GLAN research reported that small LLMs (e.g., Mistral 7B) tuned with this synthesized data showed excellent performance across mathematical reasoning, coding, academic tests, logical reasoning, and general instruction following. Remarkably, this model achieved high performance **without any actual human data for each task** (for example, Mistral 7B model reached 65% of GPT-4's level on math problem benchmarks). Additionally, GLAN's **scalability** is significant - if you want to handle new fields or concepts, you can simply add nodes to the knowledge tree to easily synthesize data for those parts.

In summary, the Seedless approach is **a method of having models create learning problems by presenting them with virtual "curricula"**. Thanks to this systematic method, it overcomes biases inherent in existing Self-Instruct techniques and can obtain synthetic datasets on very broad topics.

For detailed GLAN implementation, refer to [this GitHub repository](https://github.com/daekeun-ml/synthetic-qa-generation/tree/main/glan-instruct).

### 2. Techniques for Generalized Instruction Tuning (GLAN Case Study)

As seen from GLAN described above, the core of seedless synthesis is **generalization** and **systematization**. Looking at specific implementation methods and their effects through the GLAN case study in more detail:

* **Comparison with Prior Research:** Previously, methods like FLAN, Self-Instruct, WizardLM Evolve-Instruct relied on some **seed examples** or **human-written guidelines** to create data. For instance, Self-Instruct had humans directly create a few questions/answers, then had models imitate them to generate additional data, while WizardLM prepared relatively few easy prompts and had models gradually transform them into more complex prompts. These approaches had limitations where **fields not in seeds or new types of tasks couldn't be generated**. In contrast, GLAN uses the human knowledge map as input from the beginning, enabling creation of **data so comprehensive that almost no topics are excluded**.
* **Systematic Difficulty Control:** As seen in GLAN procedures, within one subject, **difficulty can be controlled** from problems asking only basic concepts to problems interweaving multiple concepts. This has advantages for models to undergo **step-by-step learning** or be exposed to everything from easy to difficult. While Evolve-Instruct techniques also aimed for difficulty increases, GLAN is more structural in creating easy→difficult hierarchies by initially planning curricula.
* **Comprehensive Ability Improvement:** Looking at model performance (GLAN-Mistral7B etc. in the paper) trained with GLAN-generated data, it showed **uniform improvement across various benchmarks** including mathematical reasoning, code writing, academic tests (MMLU), logic puzzles, and general sentence generation. This means performance improved as a **generalist model** overall, not just for specific few tasks. This proved that seedless methods can sufficiently develop models' capabilities across all areas. While still not reaching super-large model levels like GPT-4, it has value as a strategy for creating **significantly improved second-tier models**.
* **Customization Ease:** One of GLAN's major advantages is **ease of user customization**. For example, when tuning models for specific corporate domains, you can strengthen or add relevant field nodes in the general knowledge tree and generate more synthetic data for those parts. By additionally fine-tuning with only this generated data, you can **reinforce specific field capabilities while maintaining existing general knowledge**. Conversely, if certain areas aren't needed, you can adjust by excluding that data from training.
* **Limitations and Supplements:** Seedless approaches also have challenges. Having models generate data completely alone can cause **distortions in factuality and naturalness**. There are risks of patterns slightly different from human questioning or model knowledge errors mixing into answers. To supplement this, **human review** or **partial seed usage** are often combined. For example, there were attempts to use Persona Hub's zero-shot method while providing some few-shot actual question data as examples to reduce awkward questions. While fully automatic data generation is ultimately ideal, providing small amounts of high-quality greetings or format examples as seeds when necessary is also a practical compromise.

In summary, seedless synthetic data generation is an approach that obtains data using only **models' internal knowledge and structured prompt strategies**, and its **effectiveness and versatility** are already being proven through research. Using methodologies like GLAN, you can automatically generate data that mimics vast human expert knowledge systems to tune models, providing powerful solutions even in new domains or low-resource environments.

While specific implementation code isn't provided here, refer to the code snippet below for a quick start.

```python
import asyncio
import random
from typing import List, Dict

# Self-Instruct based data generation
class SelfInstructGenerator:
    def __init__(self):
        self.task_types = ['classification', 'generation', 'qa', 'reasoning']
        self.seed_templates = [
            "Classify {item} into categories: {categories}",
            "Generate {content} about {topic}",
            "Answer: {question}",
            "Analyze {scenario} and explain {aspect}"
        ]
    
    async def generate_instructions(self, count: int = 100) -> List[Dict]:
        """Generate self-instructions"""
        instructions = []
        
        for i in range(count):
            template = random.choice(self.seed_templates)
            task_type = random.choice(self.task_types)
            
            # Generate new instruction with LLM
            new_instruction = await self._create_instruction(template, task_type)
            
            # Generate input-output pairs
            io_pair = await self._generate_io_pair(new_instruction)
            
            instructions.append({
                'instruction': new_instruction,
                'input': io_pair['input'],
                'output': io_pair['output'],
                'task_type': task_type
            })
        
        return instructions
    
    async def _create_instruction(self, template: str, task_type: str) -> str:
        prompt = f"Create a new {task_type} instruction based on: {template}"
        # AWS Bedrock call
        return await self._call_bedrock(prompt)
    
    async def _generate_io_pair(self, instruction: str) -> Dict:
        prompt = f"For instruction '{instruction}', create realistic input and output"
        response = await self._call_bedrock(prompt)
        
        # Parsing logic
        parts = response.split("Output:")
        return {
            'input': parts[0].replace("Input:", "").strip(),
            'output': parts[1].strip() if len(parts) > 1 else ""
        }


# Diversity assurance
class DiversityController:
    def __init__(self):
        self.dimensions = {
            'complexity': ['simple', 'moderate', 'complex'],
            'format': ['text', 'list', 'json', 'code'],
            'domain': ['general', 'technical', 'creative', 'analytical']
        }
    
    def ensure_diversity(self, instructions: List[Dict]) -> List[Dict]:
        """Ensure diversity"""
        diverse_set = []
        used_combinations = set()
        
        for instruction in instructions:
            combo = self._get_combination(instruction)
            if combo not in used_combinations:
                diverse_set.append(instruction)
                used_combinations.add(combo)
        
        return diverse_set

# Domain-specific knowledge generation
class KnowledgeGenerator:
    def __init__(self):
        self.knowledge_types = ['factual', 'procedural', 'conceptual']
    
    async def generate_domain_knowledge(self, domain: str) -> List[Dict]:
        """Generate domain knowledge"""
        knowledge_items = []
        
        for knowledge_type in self.knowledge_types:
            prompt = f"""
Generate {knowledge_type} knowledge for {domain} domain.
Include:
1. Core concepts
2. Key relationships  
3. Practical applications
4. Common misconceptions
"""
            
            response = await self._call_bedrock(prompt)
            knowledge_items.append({
                'domain': domain,
                'type': knowledge_type,
                'content': response
            })
        
        return knowledge_items

# Creative content generation
class CreativeGenerator:
    def __init__(self):
        self.techniques = ['analogical', 'constraint_based', 'combinatorial']
    
    async def generate_creative_content(self, content_type: str) -> List[Dict]:
        """Generate creative content"""
        creative_items = []
        
        for technique in self.techniques:
            if technique == 'analogical':
                prompt = f"Create {content_type} using analogy between nature and technology"
            elif technique == 'constraint_based':
                prompt = f"Create {content_type} with constraint: use exactly 5 colors"
            else:  # combinatorial
                prompt = f"Create {content_type} combining music and mathematics"
            
            content = await self._call_bedrock(prompt)
            creative_items.append({
                'content': content,
                'technique': technique,
                'type': content_type
            })
        
        return creative_items
```

### 3. Seedless Synthetic Data Generation Architecture

This is an architecture for generating synthetic data using pure LLM capabilities without existing data. It processes automatic instruction generation through Self-Instruct, domain knowledge extraction, and creative content generation using Lambda and Bedrock. It collects data in real-time with Kinesis, manages quality with Step Functions, and integrates data with Glue for developing new domain AI models or general instruction following models.

The data generation process in the seedless approach proceeds through a 7-stage automated pipeline. In the first stage, AWS Lambda functions automatically generate various types of task instructions including classification, generation, Q\&A, and reasoning based on basic templates. Simultaneously, domain-specific expert knowledge extraction processes proceed, systematically generating and structuring factual, procedural, and conceptual knowledge in specific fields like healthcare, legal, and finance.

In the third stage, creative content like novels, poetry, and scenarios is generated by applying creative techniques such as analogical, constraint-based, and combinatorial thinking. All generated data is collected in real-time through Amazon Kinesis Data Streams for streaming processing.

The collected data undergoes multi-stage quality verification through complex workflows managed by AWS Step Functions. This process comprehensively evaluates language quality, content consistency, and creativity. AWS Glue integrates data generated from various sources and performs refinement tasks like duplicate removal and format unification. Finally, all high-quality synthetic data that passes verification is stored in Amazon S3 and registered in catalogs with metadata for future utilization as assets.

<figure><img src="../../.gitbook/assets/seedless_architecture.png" alt=""><figcaption></figcaption></figure>

_Figure 2. Seedless Synthetic Data Generation Architecture_

> Note: This architecture represents a reference example for illustration purposes and may not reflect all production requirements. Actual implementations may be simpler or more complex depending on specific use cases, scale requirements, and organizational constraints. Always consider your unique requirements when adapting this pattern.

### Architecture Overview

#### 3.1. Self-Instruct Engine

* AWS Lambda - Instruction generation
  * **Automatic instruction generation**: Automatically generate new instructions from basic templates
  * **Diversity assurance**: Various task types including classification, generation, QA, reasoning
  * **Scalability**: Serverless for large-scale processing
* Amazon Bedrock - LLM generator
  * **Self-Instruct execution**: Generate input-output pairs based on generated instructions
  * **Model selection**: Utilize various LLM models like Claude, Llama
  * **Batch processing**: Efficient processing of large numbers of instructions

#### 3.2. Knowledge Generation

* AWS Lambda - Knowledge extraction
  * **Domain knowledge generation**: Automatically generate expert knowledge in specific fields
  * **Knowledge type classification**: Distinguish factual, procedural, conceptual knowledge
  * **Quality control**: Verify consistency and accuracy of generated knowledge
* Amazon Bedrock - Expert knowledge LLM
  * **Domain specialization**: Generate knowledge in specialized fields like healthcare, legal, finance
  * **Multi-layer reasoning**: Step-by-step explanation of complex concepts
  * **Interconnectedness**: Express relationships and dependencies between concepts

#### 3.3. Creative Generation

*   AWS Lambda - Creative engine

    * **Creative technique application**: Analogical, constraint-based, combinatorial approaches
    * **Diversity enhancement**: Generate unpredictable creative results
    * **Quality balance**: Maintain balance between creativity and practicality
    * **Creative technique implementation:**

    ```python
    class CreativeEngine:
        def __init__(self):
            self.techniques = {
                'analogical': self.analogical_thinking,
                'constraint_based': self.constraint_based_thinking,
                'combinatorial': self.combinatorial_thinking
            }
        
        def analogical_thinking(self, topic):
            return f"Please explain {topic} by comparing it to natural phenomena"
        
        def constraint_based_thinking(self, topic):
            constraints = ["use only 5 colors", "within 100 characters", "match rhythm"]
            constraint = random.choice(constraints)
            return f"Please write about {topic} with the constraint: {constraint}"
        
        def combinatorial_thinking(self, topic):
            domains = ["music", "mathematics", "cooking", "sports"]
            domain = random.choice(domains)
            return f"Present new ideas by combining {topic} with {domain}"
    ```
* Amazon Bedrock - Creative LLM
  * **Creative content**: Generate novels, poetry, dialogue, scenarios
  * **Style diversity**: Support various writing styles and genres
  * **Emotional expression**: Natural content rich in emotions

#### 3.4. Quality Management

* AWS Step Functions - Workflow
  * **Complex quality management**: Coordinate multi-stage verification processes
  * **Conditional execution**: Branch processing based on quality scores
  * **Error handling**: Retry and alternative paths on failure
* Amazon SageMaker - Quality filter
  * **Automatic quality evaluation**: Calculate quality scores based on ML models
  * **Multi-dimensional evaluation**: Comprehensive evaluation of consistency, creativity, practicality
  * **Learning improvement**: Improve generation models through quality evaluation results
* Amazon Bedrock - Evaluation model
  * **Language quality**: Evaluate linguistic quality including grammar, vocabulary, style
  * **Sentiment analysis**: Verify emotional appropriateness of generated content
  * **Topic consistency**: Evaluate alignment between topics and content

#### 3.5. Data Integration

* Amazon Kinesis Data Streams - Real-time collection
  * **Streaming processing**: Real-time collection of generated data
  * **Scalability**: Automatic scaling with increasing data volume
  * **Durability**: Replication and backup to prevent data loss
* Amazon S3 - Staging
  * **Temporary storage**: Temporary storage for data being processed
  * **Batch processing**: Batch processing after accumulating certain amounts
  * **Data partitioning**: Efficient structure by date and type
* AWS Glue - Data integration
  * **ETL processing**: Integrate and transform data from various sources
  * **Schema evolution**: Automatic response to data structure changes
  * **Data quality**: Duplicate removal, format unification, validation

#### 3.6. Result Storage

* Amazon S3 - Synthetic data
  * **Large-scale storage**: Store large amounts of generated synthetic data
  * **Tiered storage**: Automatic storage class adjustment based on access frequency
  * **Version management**: Track data change history

Through this architecture, high-quality diverse synthetic data can be generated without existing data and utilized for new domain or creative AI model development.

Finally, we'll address **common considerations and techniques when creating synthetic data**, whether seed-based or seedless (e.g., additional data augmentation, personal information removal, data verification, red team testing, version management, etc.).
