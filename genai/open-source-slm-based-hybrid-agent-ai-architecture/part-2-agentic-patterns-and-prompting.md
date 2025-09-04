# Part 2: Agentic Patterns & Prompting

#### Content Level: 200

## Suggested Pre-Reading

* [Part 1: Overview & Background](part-1-overview-and-background.md)

### TL;DR

This is a hybrid pattern that utilizes SLM as the primary processing agent, falling back to Claude in unreliable or complex situations. Claude handles meta-roles such as query rewriting, planning, and quality verification, and achieves parallel processing and specialized task distribution through multi-agent orchestration. For SLM tooling, few-shot prompting is more effective than zero-shot prompting.

### 1. Agentic Patterns

<figure><img src="../../.gitbook/assets/slm_llm_hybrid_part2.png" alt=""><figcaption></figcaption></figure>

_Figure 1. Core Mechanisms of the Open Source SLM-Based Hybrid AI Architecture_

This diagram illustrates the core mechanisms of the open source SLM-based hybrid agent AI architecture:

* **SLM Agent**: The primary agent that processes user queries.
* **Confidence Check**: Verifies the reliability of SLM responses and triggers fallback if necessary.
* **Fallback to LLM**: Delegates complex queries or failures to the large language model Claude.
* **LLM**: Performs various tasks such as query rewriting, context summarization, routing/planning, and response evaluation.
* **Agent Orchestration**: A multi-agent structure that processes tasks in parallel or sequentially.

This allows the SLM-centric hybrid architecture to achieve reliability and scalability in various ways.

#### 1.1. SLM-First Agent Execution

When constructing enterprise chatbots or knowledge exploration agents, adopting open source SLMs as the primary model can dramatically reduce latency and costs. Small models have fewer parameters, enabling faster inference speeds, and when self-hosted, incur no API call costs, making them economical for high-volume traffic processing. For example, fine-tuning a 7B-scale LLaMA family model with quality internal FAQ data can provide response accuracy comparable to hundred-billion-scale models while delivering stable real-time responses on proprietary servers. According to research, well-fine-tuned billion-scale models for specific tasks sometimes perform equally to much larger GPT-3 models – experiments showed 1,400 times smaller models achieving similar results to GPT-3, proving the efficiency of lightweight specialized models.\[^2] Thus, deploying multiple small models in parallel/dedicated roles as needed can be more cost-effective than overworking a single giant model.

The advantages of the SLM-first strategy are summarized as follows:

* **Low Cost and High Throughput**: Open models can operate without API fees and handle high QPS (Queries per Second) by deploying multiple instances for horizontal scaling. Self-hosting open source SLMs has been reported to achieve cost savings of up to 5 times or more compared to cloud API usage.\[^1]
* **Latency Stability**: While closed APIs may experience latency variations or temporary failures during calls, self-hosted SLMs can guarantee consistent response times through controllable infrastructure. GPU/accelerator optimization can further reduce latency, and auto-scaling enables load response.
* **Data Privacy and Security**: Since models operate on internal servers, customer/enterprise data need not be transmitted externally. This is a very important advantage in strictly regulated industries like finance and healthcare, where on-premises LLMs can satisfy sensitive information protection and compliance requirements.
* **Easy Model Customization**: Open models allow acquiring weights and fine-tuning with proprietary data or applying additional controls, enabling injection of enterprise-specific tone, knowledge, and functionality. Conversely, API models can only rely on prompt engineering without internal adjustments. The transparency of open models also makes debugging unpredictable outputs or correcting biases easier.

Of course, SLMs are not always omnipotent, and overly small models may miss question **nuances** or make mistakes in **tool usage**. Additionally, the ability to process very extensive contexts like internal documents or problems requiring broad common sense/reasoning still favor the latest large models.\[^3]

Therefore, a mechanism is needed where **low-difficulty and repetitive tasks are quickly processed by SLMs** while **complex problems are delegated to superior models**. The next section addresses such **confidence-based fallback** strategies.

#### 1.2. Confidence-Based Fallback to Claude

Even when primarily utilizing SLMs, the system must be able to recognize uncertain situations and seek help from superior models to maintain high reliability. In this architecture, when SLM agents detect failure signals during answer generation, they escalate the request to large language models like Anthropic Claude. Failure signals can include the following cases:

* **Tool Usage Failures**: When tools called by SLMs produce errors or results cannot be interpreted (e.g., API timeouts, code execution errors, "file not found", etc.).
* **Output Format/Grammar Errors**: When code or SQL queries generated by SLMs have syntax errors or other obvious execution errors.
* **User Feedback**: When users indicate incorrect answers with responses like "that's wrong" or "try again". This suggests the model misunderstood context or missed information, serving as a signal for superior model intervention.
* **Ambiguous Queries/Ambiguity**: When questions themselves are vague or can be interpreted multiple ways. If SLMs judge they are confused about question intent, they can request re-questioning from Claude or have Claude generate clarifying questions to reconfirm intent with users.
* **Sensitive or High-Risk Responses**: For fields like medicine and law where incorrect answers have significant impact, or queries on topics sensitive according to company policy. In such cases, having regulated RLHF models like Claude respond reduces risk rather than small models. Also when external knowledge grounding risk is high with potential hallucination, superior models are assigned to ensure accuracy.

When fallback is triggered, Claude performs **support missions** appropriate to its role in the situation. For example, if SLM code generation repeatedly fails, Claude can directly provide correct code examples. If queries are ambiguous, Claude can clearly restate conversation intent or ask for necessary additional information. Claude possesses extensive **prior knowledge** and **advanced reasoning** capabilities, serving as a **safety net** to fill gaps SLMs cannot resolve.

After Claude is temporarily deployed through fallback and problems stabilize, **transferring control back to SLMs** is a characteristic of this system. For example, in Goose framework's lead/worker mode, when small models (workers) fail **N consecutive times**, large models (leads) temporarily intervene, then return work to workers after resolving situations. Similarly, in our architecture, after Claude establishes answer frameworks or solves difficult problems, detailed response composition or additional user conversations are resumed by SLMs.\[^4] This achieves **maximum performance at minimum cost**.

From another perspective, Claude also performs a **quality assurance (QA)** role. When SLM draft answers are produced, Claude can **evaluate/critique** them (e.g., asking "Are there any errors or risk factors in this answer?") to provide feedback for SLM self-improvement. Research shows that introducing such **self-reflection loops** improves model problem-solving capabilities.\[^5] In Anthropic's multi-Claude agent experiments, **separate Claude judge models** evaluated conclusions from other agents, scoring accuracy, source matching, etc., and requiring corrections when problems existed, thereby improving final performance. In our architecture too, Claude reviews SLM responses and corrects when necessary, producing **more stable and reliable** responses than SLMs alone.

In summary, having a **dual safety net** of "fallback to commercial LLM on failure" above the SLM-first system ensures that no matter how complex or exceptional user queries are, they are ultimately completed through large model capabilities.

#### 1.3. Claude-Assisted Query Rewriting, Summarization, and Meta-Reasoning

In hybrid architectures, commercial LLMs don't just **substitute answers** for difficult questions, but perform various **meta-functions** from **query interpretation** to **answer verification**. Key utilization patterns include:

* **Query Rewriting**: When user input is ambiguous or verbose, Claude transforms it into clear, model-friendly versions. For example, Claude can internally paraphrase ambiguous questions into multiple interpretations then select the most appropriate interpretation. Or it can summarize users' long descriptive questions to make them easier for SLMs to process. This helps SLMs focus on core questions without intent recognition errors. Claude's superior language understanding is utilized like a router.
* **Context Summarization**: When user conversation contexts or documents retrieved through search are very extensive, Claude summarizes/organizes relevant information to provide to SLMs. For example, when users upload multiple documents and ask questions, Claude summarizes each document or extracts only parts related to queries to deliver to SLMs. This enables small models with limited context windows to generate answers reflecting the essence of long contexts. Since Claude 4 Sonnet models can process 1M tokens, structures where Claude first compresses vast information from enterprise databases or reports for SLM utilization are possible. This is a frequently used pattern in RAG pipelines, where large models function as long-text summarizers.
* **Dynamic Routing and Planning**: As described earlier, Claude can decide what procedures and tools are needed according to queries and formulate plans. Compared to small models, Claude excels at logical planning, appropriately decomposing complex problems. For example, when complex requests like "Tell me about market reactions to our company's Product A and competitor trends" arrive, Claude can divide this into steps like "1) Collect social media sentiment about Product A, 2) Research competitor new product announcements, 3) Integrate both information to derive insights." Then it performs routing to assign appropriate specialized agents/tools to each step. Here Claude acts as a conductor rather than performing all tasks itself, writing instructions (prompts) for SLM sub-agents to execute. Similar to Anthropic cases where lead Claude formulated plans and summoned sub-agents with focused goals for parallel exploration. Claude's meta-understanding and control capabilities enable effective multi-agent collaboration.
* **Reflection and Critique**: Claude operates as a critic for intermediate outputs or final answers from SLMs or other agents. For example, asking Claude "What are the weaknesses in the above answer?" or "What parts need additional verification?" helps identify inaccurate or logically insufficient parts in answers. Claude's feedback is used as triggers for SLMs to improve answers or take additional measures.

In summary, Claude serves more as an advisor thinking intelligently at higher levels rather than a simple response generator. Such non-generative utilization of Claude becomes a central safety mechanism and productivity enhancement factor in small model-centered systems. Rather than positioning Claude as an omnipotent problem-solver, positioning it as a planning/critique specialist AI improves overall system efficiency.

#### 1.4. Multi-Agent Orchestration Patterns

When single agents have limitations in solving complex problems, **multi-agent architectures** that maximize performance through **collaboration of multiple LLM agents** shine. Key patterns include:

* **Orchestrator-Worker Pattern**: A structure where a central coordinator (orchestrator) agent exists and multiple worker agents perform actual tasks. Generally, the orchestrator handles high-level planning and integration roles (in our case, this role is handled by SLMs or Claude when necessary), while workers are SLM instances optimized for individual tasks. The orchestrator receives user queries, divides tasks, and decides which worker/tool to assign each to. Workers perform their assigned sub-tasks (e.g., specific question searches, database queries, text summarization) then return results to the orchestrator. Finally, the orchestrator synthesizes these to respond to users. This pattern improves overall efficiency by deploying each model to parts where they can be best utilized.
* **Parallel Sub-tasks and Token Capacity Expansion**: One major advantage of multi-agents is time reduction through parallel execution and context capacity increase. Different agents simultaneously exploring multiple information sources can collect broader facts much faster than serial processing. For example, in Anthropic experiments, web research that took a single Claude 30 steps and 10 minutes was completed in much shorter time by 10 small-scale Claude agents working in parallel. Additionally, since each agent uses separate context windows, lead + N sub-agents collaborating essentially provides N times the token context utilization effect. This is particularly useful when covering vast search spaces or handling information beyond one model's context limits. Anthropic reported that such parallel token usage explains 80% of performance improvements, achieving 90.2% higher performance than single models in actual internal tasks.
* **Serial Task Sequencing and Memory Handoff**: In some scenarios, sequential agent chains are more appropriate than parallel ones. Here, agent A's output becomes agent B's input, B's output goes to C, forming a pipeline. For example, before answering customer questions, agent1 classifies user intent, agent2 queries related internal data, then agent3 generates final answers. Each step has specialized agents, with previous step memory (conversation context or results) handed off to next agents. Central orchestrators store/manage each intermediate result and include only necessary parts in next agent prompts for such memory transfer. In multi-turn conversations, conversation history is stored in shared memory to enable long-term context maintenance. Orchestrators handle workflow control like rolling back to checkpoints on failure or replacing agents to prevent entire process interruption.
* **Expert Agents and Tool Usage**: Each agent can be optimized for specific fields or tool usage to operate like expert groups. For example, one SLM specializes in natural language summarization while another excels at SQL query generation through prompt engineering or fine-tuning for role assignment. When user questions are complex, orchestrators distribute work appropriately like "this part needs database queries so goes to SQL agent, that part needs report summarization so goes to summary agent." This allows each agent to work only in their strength areas, improving efficiency and accuracy. This structure resembles human organizational collaboration, ultimately creating effects where AI agents work as teams.

When implementing such multi-agent orchestration centered on SLMs, a key consideration is whether to make the **orchestrator** itself an SLM or a superior model like Claude. Basically, lightweight models should attempt to oversee everything, but when complexity exceeds certain levels, **hybrid** approaches where Claude takes over orchestration roles are reasonable, as discussed earlier. This can be summarized as "easy collaboration among SLMs, difficult coordination with Claude intervention." This too can be viewed as a balance point between cost and performance.

Consequently, multi-agent architectures provide paths to overcome single LLM limitations through **collective intelligence** of multiple models. In enterprise environments, utilizing such structures for broad knowledge exploration or complex task automation enables AI agents to perform **parallel and elastic problem-solving** like human teams dividing work.

### 2. Prompting Techniques for Tool Use

When utilizing open source SLMs as agents, prompt design significantly impacts performance. Particularly, Few-Shot prompt strategies providing explicit demonstration examples are effective for enabling SLMs to interact with external tools (e.g., search, calculation, DB). While large models partially learn tool usage scenarios like function calls during RLHF processes, small models lack such knowledge and need to be taught tool usage practices.

* **Zero-shot**: Giving only instructions like "Think about procedures needed to answer the following question and use search tools if necessary" without any examples. While latest LLMs show some tool usage behavior zero-shot, small models are prone to reasoning errors or format mistakes. Incorrect tool calls lead directly to failures, making stable tool usage difficult to expect zero-shot.
* **Few-shot Example Provision**: Showing best practices of tool usage by including past conversations or scenarios as examples in prompts. Directly showing models tool call formats and contexts increases probability of following same patterns in similar future situations. LangChain team experiments also reported significant improvements in complex tool selection accuracy when adding few-shot examples. Particularly when providing three examples in message format, Claude 3 model's tool selection success rate jumped from 16% zero-shot to **52%**.\[^6] Techniques dynamically selecting 3 examples with high similarity to questions were more effective than static 3 examples, and giving fewer examples in conversation format performed better than many examples at once. These results suggest that for small SLMs too, pre-training or including domain-specific tool usage samples in prompts enables much more stable correct tool calls.

The disadvantage of Few-shot prompts is consuming much of the model's context length. However, as of 2025, many open source models support long contexts of 32K or more, and including 2-3 examples is quite feasible, making the trade-off worthwhile. Additionally, fine-tuning can internalize tool usage patterns in models, eliminating need for long examples every time. Major companies like Anthropic and OpenAI also pre-fine-tune their models with function calling and tool usage capabilities to reduce developer prompt work.

Ultimately, choosing zero-shot vs few-shot strategies depends on models' prior knowledge levels and context availability. In our architecture, we inject tool usage examples through system prompts or model tuning when possible, enabling SLMs to autonomously use tools appropriately for situations. This brings us closer to true agent automation where agents make correct judgments to select necessary tools without additional user intervention.

## Further Reading

* [Part 3: Tool Integration & Fine-Tuning](part-3-tool-integration-and-fine-tuning.md)

## References

* \[^1] [S. S. et al. (2024). _Scaling Down to Scale Up: Replacing OpenAI's LLM with Open Source SLMs in Production_.](https://arxiv.org/abs/2312.14972)
* \[^2] [Monte Carlo (2023). _RAG vs. Fine Tuning: Which One to Choose the Right Method_](https://www.montecarlodata.com/blog-rag-vs-fine-tuning).
* \[^3] [Dextralabs (2025). _Open Source LLMs vs. Closed: Best Fit for Enterprise_.](https://dextralabs.com/blog/open-source-llms-vs-closed-enterprise)
* \[^4] [Block (2025). _Treating LLMs Like Tools in a Toolbox: A Multi-Model Approach to Smarter AI Agents_.](https://block.github.io/goose/blog/2025/06/16/multi-model-in-goose)
* \[^5] [Rohan Paul (2025). _Anthropic reveals multi-agent Claude research wizardry powering 90% accuracy boost_.](https://www.rohan-paul.com/p/anthropic-reveals-multi-agent-claude)
* \[^6] [LangChain (2024). _Few-shot prompting to improve tool-calling performance_.](https://blog.langchain.com/few-shot-prompting-to-improve-tool-calling-performance/)
