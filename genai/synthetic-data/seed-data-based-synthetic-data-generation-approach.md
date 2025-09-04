# Seed Data-Based Synthetic Data Generation Approach

#### Content Level: 200-300

## Suggested Pre-Reading

* [The Necessity of Synthetic Data: Core Requirements for Modern AI Development](the-necessity-of-synthetic-data.md)

### TL;DR

The seed data-based approach is a method that utilizes existing data ("seed data") to generate synthetic data specialized for specific domains or _personas_. Through this approach, models can learn from data that reflects actual domain knowledge, making it particularly useful for training **AI optimized for specific areas** such as customer service chatbots or domain-specific QA systems. First, we extract key information from the user's source data, define **task-specific personas** based on techniques from the [**Scaling Synthetic Data Creation with 1,000,000,000 Personas**](https://arxiv.org/abs/2406.20094) paper, and then use the latest LLMs to create new data based on this foundation.

**Application Scenarios:**

* **Customer Service Chatbot Training Data**: Generate diverse customer situation-specific conversation data based on company FAQs and customer inquiry history
* **Domain-Specific QA Systems**: Build question-answer datasets based on documents from specialized fields like healthcare, legal, and finance
* **Personalized Content Generation**: Create customized content reflecting user preferences and behavioral patterns
* **Industry/Job-Specific AI Assistants**: Train professional AI assistants that reflect industry-specific terminology, processes, and regulations

### 1. Seed Data Preparation and Extraction

First, we build seed data for model training from the user's raw data. Raw data can be in various formats such as PDF, TXT, CSV, images, etc., and must be converted into consistent text/QA format. This process requires different parsing techniques for each format, and documents like PDFs that **contain mixed text, images, and tables** involve complex content extraction tasks. For example, with PDF documents, text must be extracted through OCR (Optical Character Recognition), tables must be structured and converted to text, and information contained in images may require separate descriptions. As a real example, there are publicly available notebook examples that use tools like [Unstructured](https://github.com/Unstructured-IO/unstructured) to **separately extract text/tables/images from PDFs** and generate Q\&A pairs based on this. The seed data preparation stage involves:

* Identifying and extracting important content from raw data
* Structuring when necessary: for example, summarizing documents or generating **question-answer pairs** by paragraph
* Processing various formats in an integrated manner to secure **structured datasets** for model input

The result of this stage is a **seed QA dataset** that contains implicit domain knowledge. For example, you could process several company manual PDFs to create Q\&A lists, or generate frequently asked questions from product manuals. The seed dataset serves as both the foundation for subsequent synthetic data generation and as a quality standard (since seed data is usually small in volume, it's used more for providing direction rather than being used entirely for direct model training).

### 2. Task-Specific Persona Extraction (Persona Selection)

* Persona refers to virtual character settings that models perform roles with during data generation. Recent research called _"1 Billion Personas (Persona Hub)"_ proposed a **collection of 1,000,000,000 vast personas** to enable LLMs to generate data from various perspectives. Each persona is set as a character with unique knowledge, experience, interests, and occupation, and when a specific persona is assigned to an LLM, it can generate **unique results** from that perspective. This allows the **same prompt** to produce different response patterns depending on the persona, making it possible to obtain large-scale, diverse, and rich synthetic data. The Persona Hub research automatically built these personas from web data to create synthetic data covering almost all perspectives, effectively synthesizing datasets ranging from logic/math problems to knowledge responses and game NPC dialogue.

However, when actually fine-tuning our model, we cannot **utilize all personas**. Unnecessarily vast personas can actually become noise, and the fine-tuning data scale can become excessive. Therefore, we need a stage to **select or extract personas specialized for our target task**. The methods are as follows:

* **Domain Analysis:** Review seed data and target domains to identify what types of users/experts/characters are relevant. For example, in the financial domain, personas like "investment expert," "novice investor," "financial analyst" could be relevant. For customer support chatbots, role settings like "angry customer," "friendly agent" would be helpful.
* **Persona Source Utilization:** If there are vast persona repositories like Persona Hub, **filter by keywords or domains** to find relevant personas. According to research, personas in Persona Hub each contain parts of global knowledge, so selecting appropriate personas can guide models to draw out knowledge in that field. For instance, when creating medical domain QA, you could select personas like "internal medicine doctor," "patient," "pharmacist" to generate conversations.
* **Text-Based Inference:** If separate persona data is unavailable, you can infer **potential personas** from the seed data itself. For example, by examining seed Q\&A to infer who the questioner is (customer or expert) and who the respondent is (manual or person), you can define persona settings that match. Alternatively, you can ask LLMs to "analyze the tone or perspective of this text" to reverse-engineer personas.

Through **task-specific persona selection**, the tone, perspective, and expertise level of synthetic data that the model will generate becomes concrete. The important point is that selected personas must have **high relevance to the target field**. As the Persona Hub research stated, "all use cases can be connected to specific personas," any task can draw out related knowledge through appropriate persona role-play by LLMs. However, irrelevant personas (e.g., chef persona for financial data generation) should be avoided, and balance should be maintained between **persona diversity** and **realism**.

Below is a code snippet for generating task-specific personas.

```python
import random
from typing import List, Dict

class PersonaGenerator:
    def __init__(self):
        self.persona_templates = {
            'professional': [
                "A {experience_level} {profession} working in {industry}",
                "An expert {profession} with {years} years of experience in {specialty}",
                "A {profession} specializing in {domain} at a {company_size} company"
            ],
            'demographic': [
                "A {age_group} {gender} from {location}",
                "Someone with {education_level} education background",
                "A {lifestyle} person interested in {interests}"
            ],
            'behavioral': [
                "Someone who prefers {communication_style} communication",
                "A {personality_trait} individual who values {values}",
                "A person with {technical_proficiency} technical skills"
            ]
        }
        
        self.attributes = {
            'experience_level': ['junior', 'mid-level', 'senior', 'expert'],
            'profession': ['doctor', 'lawyer', 'engineer', 'teacher', 'consultant'],
            'industry': ['healthcare', 'technology', 'finance', 'education', 'retail'],
            'years': ['2-5', '5-10', '10-15', '15+'],
            'specialty': ['data analysis', 'customer service', 'project management'],
            'company_size': ['startup', 'mid-size', 'enterprise'],
            'age_group': ['young adult', 'middle-aged', 'senior'],
            'gender': ['male', 'female', 'non-binary'],
            'location': ['urban', 'suburban', 'rural'],
            'education_level': ['high school', 'bachelor\'s', 'master\'s', 'PhD'],
            'communication_style': ['direct', 'diplomatic', 'casual', 'formal'],
            'personality_trait': ['analytical', 'creative', 'detail-oriented', 'big-picture'],
            'technical_proficiency': ['basic', 'intermediate', 'advanced', 'expert']
        }
    
    def generate_task_specific_personas(self, task_domain: str, 
                                      num_personas: int = 100) -> List[Dict]:
        """Generate personas specialized for task domain"""
        personas = []
        
        for _ in range(num_personas):
            persona = self._create_single_persona(task_domain)
            personas.append(persona)
        
        return personas
    
    def _create_single_persona(self, task_domain: str) -> Dict:
        """Generate single persona"""
        # Select domain-specific attributes
        domain_specific_attrs = self._get_domain_attributes(task_domain)
        
        # Random attribute combination
        selected_attrs = {}
        for attr_type, options in domain_specific_attrs.items():
            selected_attrs[attr_type] = random.choice(options)
        
        # Generate persona text
        template_type = random.choice(list(self.persona_templates.keys()))
        template = random.choice(self.persona_templates[template_type])
        
        try:
            persona_text = template.format(**selected_attrs)
        except KeyError:
            # Generate default persona if required attributes are missing
            persona_text = f"A professional working in {task_domain}"
        
        return {
            'id': f"persona_{random.randint(10000, 99999)}",
            'description': persona_text,
            'attributes': selected_attrs,
            'domain': task_domain
        }
    
    def _get_domain_attributes(self, domain: str) -> Dict:
        """Return domain-specific attributes"""
        base_attrs = dict(self.attributes)
        
        if domain == 'healthcare':
            base_attrs['profession'] = ['doctor', 'nurse', 'pharmacist', 'therapist']
            base_attrs['specialty'] = ['cardiology', 'pediatrics', 'surgery', 'psychiatry']
        elif domain == 'technology':
            base_attrs['profession'] = ['software engineer', 'data scientist', 'product manager']
            base_attrs['specialty'] = ['machine learning', 'web development', 'cybersecurity']
        elif domain == 'finance':
            base_attrs['profession'] = ['financial advisor', 'analyst', 'accountant']
            base_attrs['specialty'] = ['investment', 'risk management', 'taxation']
        
        return base_attrs
```

### 3. Persona-Based Synthetic Data Generation (Using LLMs)

Now, based on the prepared seed data and selected persona information, we prompt LLMs to generate new synthetic data. It's recommended to use the currently available highest-performing LLM models or APIs (e.g., Claude Sonnet), and internally, prompts must be designed so that the LLM properly **references** seed data and **performs roles** as personas.

**Prompt Design:** Create prompt templates according to the type of data to be synthesized. For example, if you want to create Q\&A data:

* Prompt example: _"Referring to the following information, create question-answer pairs as \[Persona: {medical expert}]. Information: "{seed paragraph}""_
* Or _"You are {persona description}. Read the above material and generate one question that a user might ask, then provide an answer to it."_

Below is a more specific prompt example.

```python
class PersonaPromptGenerator:
    def __init__(self):
        self.prompt_templates = {
            'question_generation': """
You are {persona_description}.

Based on your background and expertise, generate {num_questions} realistic questions 
that someone like you would ask about {topic}.

Consider your:
- Professional experience and knowledge level
- Communication style and preferences  
- Specific interests and concerns related to {topic}

Format each question clearly and make them diverse in complexity and focus.
""",
            
            'response_generation': """
You are {persona_description}.

A user has asked: "{question}"

Respond as this persona would, considering:
- Your professional background and expertise level
- Your communication style and tone
- Your specific perspective on this topic
- Any domain-specific knowledge you would have

Provide a helpful, authentic response that reflects your persona's characteristics.
""",
            
            'conversation_simulation': """
You are {persona_description}.

Engage in a natural conversation about {topic}. 

Your conversation partner is: {partner_persona}

Start the conversation with a question or comment that your persona would naturally make.
Keep your responses authentic to your background and communication style.
"""
        }
    
    def generate_persona_prompt(self, persona: Dict, 
                              prompt_type: str, **kwargs) -> str:
        """Generate persona-based prompts"""
        template = self.prompt_templates.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        prompt_vars = {
            'persona_description': persona['description'],
            **kwargs
        }
        
        return template.format(**prompt_vars)
```

You can input **seed data content** or just put **keywords** extracted from seed data and ask to create questions. When persona profiles (e.g., "45-year-old experienced internal medicine doctor") are specified in prompts, LLMs will impersonate that personality to generate answers. This allows extracting _various questions and answers_ from the same seed information.

**Generation Process:** You can show LLMs one or two seed examples as **few-shot examples**, then guide them to generate answers for new inputs. Alternatively, you can ask zero-shot questions like "What questions would persona X ask about this topic?" without seeds. The important thing is **sufficient diversity** and **domain appropriateness**. According to Persona Hub research, even in generating over 100 million conversations, varying personas produced diverse outputs, and particularly utilizing LLMs' _"role-play"_ capabilities allows combining persona techniques with **almost all data generation scenarios**.

**Quality management** of generated data should also be conducted in parallel during this stage. Since LLMs sometimes provide incorrect answers or create questions that don't fit the context, **consistency verification** against seed data is performed. A simple method is to have LLMs re-examine their own generated Q\&A (refer to "Data Reliability Verification" detailed later). While excluding obviously inappropriate results, adjust to collect **rich expressions and various difficulty levels** of Q\&A.

**Example:** For instance, if synthesizing data for "customer service chatbot" purposes, prepare a few actual customer inquiry cases as seed data. Then set personas like "angry customer," "polite customer," "customer asking detailed questions" to create questions with different tones and perspectives, and have answers written by "friendly agent" persona. This diversifies question tones and makes answers carry situation-appropriate nuances. Consequently, starting from a few seed data points, you can obtain synthetic conversation data for **various customer scenarios** to broadly train chatbots.

**Output Data Format:** The final synthetic data should be in the **format** defined in the initial planning stage. Whether Q\&A pairs, conversations, summaries, etc., anything is possible, but it should be **structured** for direct use as model training input. For example, create "prompt" and "response" columns in JSON or CSV format, or organize in text format suitable for fine-tuning scripts.

This concludes our examination of the persona-specific synthetic data generation process based on seed data. To summarize the key points: **create seeds from source data → define personas suitable for tasks → grow diverse new data with LLMs**. Next, we'll look at methods for creating synthetic data without separate seeds, namely the _seedless_ approach. Common considerations for each approach (additional data augmentation, PII removal, verification, etc.) will be organized separately later.

### 4. Seed Data-Based Synthetic Data Generation Architecture

This is an architecture for generating persona-specific synthetic data utilizing existing data (PDF, CSV, text).

The entire data generation process consists of a continuous pipeline of 6 stages. First, various forms of source data are collected in Amazon S3, serving as a central repository. The collected data undergoes preprocessing where text is extracted through Amazon Textract and personally identifiable information (PII) is automatically removed using Amazon Comprehend.

The refined data is converted into question-answer format through AWS Glue to build a structured seed dataset. Subsequently, AWS Lambda functions analyze the content and context of seed data to automatically select and match the most suitable personas. The selected persona information is passed to Amazon Bedrock to generate high-quality synthetic data reflecting persona characteristics.

The generated data is evaluated for quality based on various criteria such as accuracy, completeness, and relevance through Amazon SageMaker's LLM-as-Judge functionality. Finally, data that passes quality verification is stored in Amazon S3 and registered in AWS Glue Data Catalog with metadata to facilitate future search and utilization as assets.

<figure><img src="../../.gitbook/assets/seed_architecture.png" alt=""><figcaption></figcaption></figure>

_Figure 1. Seed-based Synthetic Data Generation Architecture_

> Note: This architecture represents a reference example for illustration purposes and may not reflect all production requirements. Actual implementations may be simpler or more complex depending on specific use cases, scale requirements, and organizational constraints. Always consider your unique requirements when adapting this pattern.

#### 4.1. Raw Data Sources

* Amazon S3 - Store various forms of source data
  * **PDF Documents**: Manuals, reports, contracts, and other unstructured documents
  * **CSV Data**: Structured tabular data
  * **Text Files**: Logs, conversation records, documents, etc.

#### 4.2. Data Preprocessing

* Amazon Textract - Document parsing
  * **OCR Functionality**: Extract text from scanned documents
  * **Structure Recognition**: Automatically recognize tables, forms, key-value pairs
  * **Multi-language Support**: Process various languages including Korean and English
* Amazon Comprehend - PII Removal
  * **Automatic PII Detection**: Identify personal information like names, addresses, phone numbers, emails
  * **Confidence Scores**: Provide accuracy for each PII item
  * **Masking Processing**: Automatically mask or remove detected PII

#### 4.3. Seed Data Construction

* AWS Glue - Data transformation
  * **ETL Jobs**: Convert extracted text into Q\&A format
  * **Data Cleansing**: Remove duplicates, unify formats, verify quality
  * **Schema Evolution**: Automatically respond to data structure changes
* Amazon S3 - Seed dataset storage
  * **Structured Storage**: Store refined data in JSON, Parquet formats
  * **Partitioning**: Efficient data structure by domain and date
  * **Compression**: Optimize storage costs

#### 4.4. Persona System

*   Amazon DynamoDB - Persona DB

    * **Persona Profiles**: Store characteristics like occupation, expertise, interests, tone
    * **Fast Queries**: Millisecond response times
    * **Scalability**: Automatic scaling as persona numbers increase
    * Persona data structure example

    ```json
    {
    "persona_id": "medical_expert_001",
    "name": "Medical Expert",
    "expertise": ["internal medicine", "diagnosis", "treatment"],
    "tone": "professional and accurate",
    "background": "Internal medicine specialist with 15 years of experience",
    "language_style": "uses medical terminology, provides accurate explanations"
    }
    ```
* AWS Lambda - Persona selection
  * **Automatic Matching**: Select personas suitable for seed data content
  * **Diversity Assurance**: Provide various perspectives on the same topic
  * **Real-time Processing**: Fast persona selection and assignment

#### 4.5. Data Generation

* Amazon Bedrock - LLM generation
  * **Various Models**: Choose from Claude, Nova, GPT-OSS, etc.
  * **Persona Application**: Reflect selected persona characteristics in prompts
  * **Batch Processing**: Efficiently generate large amounts of data
* Amazon SageMaker - Quality verification. The baseline for quality evaluation criteria is as follows:
  * **Accuracy**: Does the answer correctly respond to the question?
  * **Completeness**: Is all necessary information included?
  * **Consistency**: Are persona characteristics consistently reflected?
  * **Naturalness**: Is it as natural as if written by humans?

#### 4.6. Data Asset Management

* Amazon S3 - Synthetic data storage
  * **Version Management**: Manage generated data by version
  * **Metadata Tagging**: Store information like generation conditions, quality scores
  * **Access Control**: Manage data security with IAM policies
* AWS Glue Data Catalog - Metadata management
  * **Schema Management**: Automatically recognize and manage data structures
  * **Search Functionality**: Search data based on metadata
  * **Data Lineage**: Track data generation processes

Through this architecture, you can safely and efficiently generate high-quality persona-specific synthetic data while maximizing the value of existing data.
