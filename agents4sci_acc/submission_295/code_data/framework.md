# Accelerating Materials Discovery: A Framework for Large Language Model-Driven Catalyst Design

## Part I: Literature Review - Synthesizing Knowledge for a New Discovery Paradigm

### Section 1: The Emergence of Generative AI as a Catalyst for Scientific Innovation

The landscape of scientific research is undergoing a profound transformation, driven by the advent of artificial intelligence (AI), particularly Large Language Models (LLMs). As science grapples with escalating challenges of information overload, increasing specialization leading to disciplinary silos, and the diminishing returns of conventional research methodologies, LLMs are emerging not merely as advanced tools but as powerful agents capable of reshaping the scientific process itself.<sup>1</sup> This section establishes the foundational capabilities of LLMs, explores their evolution from passive information processors to active collaborators in discovery, and critically examines the limitations and ethical imperatives that accompany their integration into the scientific enterprise.

#### The Paradigm Shift from Information Tools to Collaborative Agents

Historically, computational tools in science have focused on automating specific, well-defined tasks such as data analysis, simulation, or information retrieval. LLMs represent a paradigm shift from this model. Trained on vast and diverse corpora, including a significant portion of the world's scientific literature, these models possess an unprecedented ability to process and synthesize cross-domain knowledge.<sup>1</sup> This capability allows them to move beyond simple data retrieval and engage in complex scientific reasoning, knowledge integration, and hypothesis generation, thereby supporting and accelerating interdisciplinary breakthroughs.<sup>1</sup>

The evolution of LLMs can be characterized as a progression from simple information tools to active collaborators and, in some cases, autonomous scientific agents.<sup>1</sup> In their role as collaborators, LLMs facilitate dialog-based scientific exploration and iterative problem-solving, engaging with human researchers in a dynamic partnership.<sup>1</sup> This interactive model contrasts sharply with the traditional, linear process of scientific inquiry, where a researcher would manually sift through literature, formulate a hypothesis in isolation, and then proceed to experimental validation. The new paradigm allows researchers to "brainstorm" with an AI partner that has access to a far broader knowledge base, potentially uncovering connections and ideas that would otherwise remain hidden.<sup>4</sup> This shift is not just an acceleration of existing workflows but a fundamental restructuring of the scientific method's epistemological foundations. The traditional, human-centric cycle of literature review, hypothesis formulation, experimentation, and analysis is being reshaped into a dynamic, interactive human-AI collaborative loop. The LLM's primary function becomes the expansion of the conceptual "search space" for the human researcher, whose role, in turn, shifts from being the sole generator of initial hypotheses to the critical evaluator, refiner, and validator of a more diverse set of AI-generated possibilities.

#### Core Capabilities: Language, Reasoning, and Knowledge Integration

The power of LLMs stems from their underlying architecture, typically based on the deep neural network model known as the Transformer.<sup>3</sup> Through large-scale, unsupervised pre-training on massive text datasets—encompassing books, articles, and web pages—these models learn intricate patterns in language, enabling them to perform sophisticated natural language processing (NLP) tasks.<sup>3</sup> Key capabilities include:

**Semantic Understanding**: LLMs develop a strong capacity for semantic parsing, allowing them to interpret the meaning of specialized terminology and analyze complex sentence structures within technical domains like biomedical research or materials science.<sup>3</sup>

**Text Generation**: They can generate high-quality, human-like text that is often difficult to distinguish from content written by a human expert, a skill applied to tasks from drafting manuscripts to summarizing research papers.<sup>4</sup>

**Knowledge Synthesis**: The generalization capability acquired from pre-training on extensive corpora allows LLMs to identify and integrate cross-domain linguistic features and knowledge.<sup>3</sup> This is particularly valuable in interdisciplinary fields where breakthroughs often occur at the intersection of different disciplines.<sup>1</sup>

**Multimodality**: The modern generation of LLMs is increasingly multimodal, capable of processing not only plain text but also other forms of scientific data such as molecular structures, protein sequences, and images, further enhancing their ability to synthesize disparate information.<sup>1</sup>

These core capabilities are being applied across the entire research lifecycle. LLMs are used to assist in writing, editing, and peer review, improving research efficiency and quality.<sup>1</sup> In the discovery phase, they enable automated literature review, knowledge integration, and hypothesis generation, directly contributing to scientific innovation.<sup>1</sup>

#### Critical Limitations and Ethical Considerations

The deployment of LLMs in science is not without significant challenges and risks that demand careful consideration. A balanced and critical perspective is essential for responsible innovation.

**Factual Inaccuracy and "Hallucinations"**: LLMs are trained to generate plausible sequences of text, not to be factually accurate. This can lead to the generation of "hallucinations"—coherent, well-written statements that are completely false or nonsensical.<sup>4</sup> The coherent nature of these responses can create a false perception of authority, leading end-users to trust incorrect information without question.<sup>4</sup> This risk is particularly acute in scientific applications where precision and accuracy are paramount.

**Encoded Biases**: LLMs are trained on data created by humans and, as such, inherit the biases present in that data. These automated biases can manifest in the model's outputs, potentially perpetuating or even amplifying existing societal or scientific prejudices.<sup>4</sup>

**Outdated Knowledge**: The knowledge of a pre-trained LLM is static and limited to the information contained in its training corpus, which can quickly become outdated in rapidly advancing scientific fields.<sup>1</sup> This necessitates the development of frameworks for continuously updating their knowledge base.

**Opacity and Lack of Transparency**: Many of the most powerful LLMs are proprietary, with their inner workings and training data remaining undisclosed.<sup>5</sup> This "black box" nature makes it difficult to fully assess their performance, understand their failure modes, or verify that their outputs are properly sourced. This opacity is fundamentally at odds with the scientific values of openness and reproducibility.<sup>5</sup>

In light of these challenges, the scientific community must establish and adhere to a set of guiding principles for the use of LLMs. Transparency is of indispensable value.<sup>5</sup> Researchers must explicitly acknowledge their reliance on LLMs in the methods sections of their publications, detailing which models were used and how. To promote reproducibility and scrutiny, prompts and their corresponding outputs should be publicly released as supplementary materials whenever feasible. As a community that values transparency, preference should be given to open-source systems that disclose their architecture and training data, holding them to the same rigorous standards as other scientific instruments.<sup>5</sup>

### Section 2: The Landscape of Catalyst Design: From Edisonian Methods to Rational Design

Catalysis is a cornerstone of the modern chemical industry, enabling the efficient production of fuels, materials, and pharmaceuticals that underpin society.<sup>8</sup> The discovery of novel catalysts has historically been a slow, resource-intensive process. This section provides an overview of the fundamental principles of catalysis, traces the evolution of discovery methodologies from empirical trial-and-error to data-driven approaches, and identifies the key bottlenecks that an LLM-driven paradigm is uniquely positioned to address.

#### Fundamentals of Catalysis

At its core, catalysis is the process of increasing the rate of a chemical reaction through the addition of a substance known as a catalyst. The catalyst participates in the reaction but is not consumed, emerging unchanged at the end of the process.<sup>8</sup> Its fundamental role is to provide an alternative reaction pathway with a lower activation energy—the minimum energy required for the reaction to proceed.<sup>8</sup> By lowering this energy barrier, the catalyst allows the reaction to occur more readily at a given temperature, thereby increasing its rate.<sup>8</sup> Catalysis is broadly categorized into three types:

**Homogeneous Catalysis**: The catalyst and reactants exist in the same phase, typically a liquid solution. This allows for high reactivity but often presents challenges in separating the catalyst from the products.<sup>8</sup>

**Heterogeneous Catalysis**: The catalyst is in a different phase from the reactants, most commonly a solid catalyst with liquid or gaseous reactants. This type is dominant in industrial processes due to the ease of catalyst recovery and product separation.<sup>8</sup>

**Biocatalysis**: This involves the use of biological catalysts, such as enzymes (proteins) or ribozymes (RNA), which exhibit exceptionally high efficiency and selectivity for specific biological reactions.<sup>8</sup>

This report focuses primarily on heterogeneous catalysis, which is central to applications like the electrocatalytic nitrate reduction reaction.

#### The Iterative Nature of Traditional Catalyst Discovery

The development of a new heterogeneous catalyst is a complex, multifaceted problem. Performance is determined by a sensitive interplay of factors, including the elemental composition, surface structure, morphology, the choice of support material, and the specific synthesis and reaction conditions.<sup>10</sup> The traditional approach to navigating this vast and complex design space has been largely empirical, relying on a combination of chemical intuition, prior knowledge, and a significant degree of trial-and-error experimentation.<sup>11</sup>

This workflow is inherently iterative.<sup>10</sup> A set of candidate materials is synthesized, tested for performance, and the results are analyzed. Based on this analysis, objectives may be adjusted, and a new set of experiments is designed. This Edisonian method, while responsible for many foundational discoveries, is slow, labor-intensive, and costly.<sup>1</sup> As the "low-hanging fruit" in catalyst design has been picked, this approach faces diminishing returns, necessitating a more systematic and predictive methodology.

#### The Rise of Computational and Data-Driven Catalysis

The limitations of purely experimental methods have driven a paradigm shift towards rational catalyst design, a strategy that combines theory, computation, and data science to guide and accelerate discovery.<sup>14</sup> Machine learning (ML) has become a particularly powerful tool in this domain.<sup>16</sup>

**Descriptor-Based Machine Learning**

The pre-LLM state-of-the-art in data-driven catalysis involves training ML models—such as decision trees, random forests, support vector machines, and neural networks—to predict catalytic performance.<sup>13</sup> These models learn quantitative structure-activity relationships by correlating a set of inputs, known as "descriptors" or "features," with a target property, such as reaction yield or selectivity.<sup>12</sup> These descriptors can be based on:

**Experimental Parameters**: Synthesis variables (e.g., precursor salts, calcination temperature), operating conditions, and catalyst properties (e.g., surface area).<sup>16</sup>

**Theoretical Properties**: Computationally derived features, often from Density Functional Theory (DFT), that capture fundamental electronic or geometric properties believed to govern catalytic activity. A classic example is the "d-band center" of a transition metal surface, which correlates with the adsorption strength of reaction intermediates.<sup>17</sup>

#### The Descriptor Bottleneck and Data Challenges

While powerful, this descriptor-based approach faces two fundamental limitations. First is the descriptor bottleneck: the performance of the ML model is critically dependent on the quality and relevance of the human-engineered features.<sup>16</sup> Defining a set of descriptors that fully captures the complex, non-linear interactions governing catalysis is exceptionally difficult. This reliance on expert-defined features inherently limits the model's predictive power to known correlations and makes it difficult to discover truly novel catalytic principles.

Second is the challenge of data scarcity and bias. High-quality experimental catalysis data is expensive and time-consuming to generate, resulting in small datasets.<sup>13</sup> Furthermore, these datasets are often heavily biased. There is a selection bias, as researchers tend to explore variations of known successful catalysts, leading to an over-representation of certain elements and an under-exploration of the vast chemical space.<sup>13</sup> There is also a class imbalance, as random experiments are far more likely to produce low-performing catalysts, making high-yield examples rare.<sup>13</sup> These data limitations make it extremely challenging to train ML models that can generalize well and extrapolate to predict the performance of truly novel compositions outside the domain of the training data.<sup>13</sup>

The emergence of LLMs offers a potential solution to these deeply rooted problems. An LLM, when fine-tuned on a corpus of scientific literature, learns not from a predefined set of numerical descriptors but from the unstructured natural language text itself. This text contains not only the data (e.g., "a catalyst of composition X gave a yield of Y%") but also the authors' crucial reasoning, their discussion of mechanisms, their hypotheses about synergistic effects, and their justifications for experimental choices. By processing this rich semantic context, the model can learn a more abstract and generalizable "chemical grammar." It can infer relationships—such as "element A is often used to enhance the stability of element B due to its oxophilicity"—that are difficult to encode in a fixed vector of numbers. This allows the LLM to reason by analogy, proposing novel compositions by substituting elements with others that it has learned, from entirely different contexts, share similar functional roles. This capability represents a significant leap beyond traditional descriptor-based models, enabling a more powerful form of extrapolation and a path to overcoming both the descriptor and extrapolation bottlenecks in catalyst discovery.

### Section 3: Large Language Models in Materials Science: Current Capabilities and Frontiers

The application of LLMs to the specialized and complex domain of materials science is a rapidly advancing frontier. While general-purpose models possess broad capabilities, their effective use in scientific discovery requires significant domain-specific adaptation. This section surveys the current landscape of LLMs in materials science and chemistry, detailing the key strategies for their specialization, their principal applications across the research workflow, and the evolution towards more robust and reliable agentic systems.

#### Domain Adaptation and Specialized Models

Off-the-shelf LLMs, despite their impressive general knowledge, often struggle with the distinct complexities of materials science, exhibiting conceptual errors, factual hallucinations, and an inability to reason over core domain principles.<sup>7</sup> Consequently, a major focus of current research is on domain adaptation—the process of refining a pre-trained model using specialized datasets to align its responses more closely with a specific field.<sup>21</sup>

This adaptation is typically achieved through fine-tuning, where a foundational LLM is further trained on a curated corpus of domain-specific text and data.<sup>21</sup> This process endows the model with a deeper understanding of the specialized vocabulary, notations (e.g., chemical formulas), and conceptual relationships of materials science. Several domain-specific LLMs have been developed, demonstrating the power of this approach. For instance,

**LLaMat** is a family of models developed by continued pre-training of LLaMA on an extensive corpus of materials literature and crystallographic data, showing superior performance in materials-specific information extraction and even crystal structure generation.<sup>23</sup> Similarly, the

**DARWIN** series of models incorporates scientific knowledge by fine-tuning on instruction data points automatically generated from scientific texts, eliminating the need for manual curation.<sup>24</sup> These efforts highlight a critical trend: the future of AI in science lies not in generalist models alone, but in a diverse ecosystem of specialized models tailored for specific scientific domains.

#### Key Applications in the Research Workflow

Domain-adapted LLMs are being deployed across a wide range of tasks to accelerate materials research and discovery.<sup>25</sup>

**Knowledge Extraction and Text Mining**: The vast majority of materials science knowledge is locked within unstructured text in millions of scientific publications. LLMs are proving to be exceptionally powerful tools for automated knowledge extraction, a critical first step in building the large, high-quality datasets needed for any data-driven discovery effort.<sup>7</sup> They can perform intricate chemical text mining tasks such as recognizing chemical compound entities, labeling the roles of reactants in a reaction, and extracting detailed synthesis information from experimental paragraphs, often outperforming models specifically designed for these tasks with minimal annotated data.<sup>28</sup>

**Material Property Prediction**: A significant frontier is the use of LLMs as predictive models. By representing materials as text strings (e.g., chemical compositions, SMILES for molecules, or natural language descriptions of crystal structures), LLMs can be fine-tuned to predict a wide array of physical and chemical properties.<sup>25</sup> Studies have shown that LLMs can achieve competitive accuracy in predicting properties like atomization energies, band gaps, and polymer adhesive energies, particularly in low-data environments where traditional ML models might struggle.<sup>32</sup> The ability to learn from textual descriptions challenges the conventional reliance on graph-based neural networks and opens new avenues for property prediction.<sup>35</sup>

**Hypothesis Generation and Materials Design**: The most transformative application is in de novo design. By synthesizing the knowledge embedded in their training data, LLMs can be prompted to generate novel hypotheses for new materials.<sup>20</sup> This capability positions LLMs within the broader field of generative AI for chemistry, which aims to create entirely new molecules and materials with desired properties from scratch.<sup>36</sup> This generative function allows researchers to explore vast, unknown regions of the chemical design space, moving beyond simple optimization of known systems.<sup>37</sup>

#### The Grounding Problem and the Rise of Agentic Systems

A fundamental challenge in using LLMs for scientific discovery is the "grounding problem"—ensuring that their outputs are not only linguistically plausible but also factually correct, physically realistic, and consistent with the laws of chemistry and physics.<sup>7</sup> To address this, the field is rapidly moving beyond monolithic LLMs towards more sophisticated, multi-component systems, often described as "agents," that integrate LLMs with reliable external resources.

**Retrieval-Augmented Generation (RAG)**: RAG is a powerful technique that grounds an LLM's responses in factual data. Before generating an answer, a RAG system retrieves relevant information from a trusted, up-to-date knowledge base (e.g., a materials database or a curated set of recent papers) and provides this information to the LLM as context within the prompt.<sup>21</sup> This significantly reduces the likelihood of hallucination and allows the LLM to incorporate knowledge beyond its static training data.<sup>40</sup>

**Tool-Augmented LLMs**: The next step in this evolution is to grant LLMs the ability to use external "tools".<sup>20</sup> A tool can be any external resource or function the LLM can call upon, such as a calculator, a Python interpreter, an API for a scientific database, or even a simulation code.<sup>7</sup> This allows the LLM to offload tasks it is not well-suited for (like precise numerical calculations) to a specialized tool, and to actively interact with its environment to gather information or perform actions. Systems like **HoneyComb** exemplify this approach by augmenting an LLM with a curated materials science knowledge base (MatSciKB) and a hub of computational tools (ToolHub), enhancing the accuracy and relevance of its outputs for specialized tasks.<sup>7</sup> This agentic framework, which combines the linguistic and reasoning capabilities of an LLM with the factual reliability of databases and the computational power of simulation tools, represents the current state-of-the-art and the most promising path forward for building practical and impactful AI systems for materials discovery.

To provide a clearer picture of the current model landscape, Table 2 summarizes key foundational and domain-specific LLMs relevant to scientific applications. This comparison is essential for justifying the selection of a base model for the research proposal that follows.

#### Table 2: Overview of Foundational and Domain-Specific LLMs for Scientific Applications

| Model Name | Base Architecture | Training Corpus | Key Features | Demonstrated Applications in Materials Science/Chemistry |
|------------|-------------------|-----------------|--------------|--------------------------------------------------------|
| GPT-4<sup>42</sup> | Transformer (Decoder) | General (Web text, books) | Instruction-tuned, Multi-modal (text, image), Proprietary | High performance on chemistry Q&A, text mining, reasoning tasks<sup>31</sup> |
| Llama 3<sup>30</sup> | Transformer (Decoder) | General (Public web data) | Instruction-tuned, Open-weights, Scalable (8B to 70B parameters) | Strong baseline for fine-tuning, competitive on text mining tasks<sup>30</sup> |
| Galactica<sup>2</sup> | Transformer (Decoder) | Scientific (Papers, reference material) | Trained on scientific text and data, can work with modalities like SMILES | Designed for scientific knowledge tasks, though faced initial issues with accuracy<sup>2</sup> |
| MatSciBERT<sup>43</sup> | Transformer (Encoder) | Materials Science Literature | Domain-specific vocabulary and pre-training on materials abstracts | Improved performance on downstream NLP tasks like named entity recognition in materials science |
| LLaMat<sup>23</sup> | LLaMA (Decoder) | Materials literature, crystallographic data | Continued pre-training on domain data, specialized CIF variant | Superior performance on materials information extraction and crystal structure generation<sup>23</sup> |
| DARWIN<sup>24</sup> | LLaMA (Decoder) | Public datasets, literature | Fine-tuned with auto-generated scientific instruction data (SIG model) | State-of-the-art results on various scientific tasks, including property prediction<sup>24</sup> |

### Section 4: Electrocatalytic Nitrate Reduction to Ammonia: A Grand Challenge in Sustainable Chemistry

To ground the potential of LLM-driven discovery in a concrete and impactful scientific problem, this report focuses on the electrocatalytic nitrate reduction reaction (NO₃RR). This reaction sits at the critical nexus of environmental remediation and sustainable chemical production, presenting a formidable challenge for catalyst design that is ripe for a new discovery paradigm. This section details the scientific and societal motivations for studying NO₃RR, outlines the complexities of its reaction chemistry, reviews the current state of catalyst development, and identifies the key scientific gaps that the proposed research aims to address.

#### Dual Motivation: Environmental Remediation and Sustainable Synthesis

The intense research interest in NO₃RR stems from its potential to simultaneously address two pressing global challenges<sup>45</sup>:

**Nitrate Pollution Remediation**: Anthropogenic activities, primarily the extensive use of nitrogen-based fertilizers in agriculture and industrial processes, have led to a severe imbalance in the global nitrogen cycle.<sup>45</sup> This has resulted in the accumulation of harmful nitrate (NO₃⁻) and nitrite (NO₂⁻) ions in groundwater and surface waters. Elevated nitrate levels cause eutrophication in aquatic ecosystems and pose significant risks to human health.<sup>45</sup> Electrocatalysis offers a promising method for remediating nitrate-contaminated water by converting the pollutant into benign or valuable products using clean electrons as the reducing agent.<sup>49</sup>

**Sustainable Ammonia Synthesis**: Ammonia (NH₃) is one of the world's most produced chemicals, essential for manufacturing fertilizers and increasingly viewed as a potential carbon-free energy carrier due to its high hydrogen density.<sup>45</sup> The current industrial production method, the Haber-Bosch process, is extremely energy-intensive and contributes significantly to global CO₂ emissions.<sup>46</sup> The NO₃RR provides an alternative, sustainable route to ammonia synthesis that can operate under ambient conditions and be powered by renewable electricity. This "waste-to-wealth" concept, turning a pollutant into a valuable commodity, is a powerful driver for innovation in this field.<sup>49</sup>

#### The Complex Reaction Network

The electrochemical reduction of nitrate is a complex process involving multiple proton and electron transfer steps. The final product depends heavily on the catalyst and reaction conditions. The two most desirable pathways are the 8-electron reduction to ammonia and the 10-electron reduction to dinitrogen gas (N₂)<sup>46</sup>:

NO₃⁻ + 9H⁺ + 8e⁻ → NH₃ + 3H₂O

2NO₃⁻ + 12H⁺ + 10e⁻ → N₂ + 6H₂O

Achieving high selectivity for a single product is a major challenge due to a complex network of possible intermediates (e.g., NO₂⁻, NO, N₂O) and competing side reactions.<sup>46</sup> The most significant competing reaction is the hydrogen evolution reaction (HER), where protons in the electrolyte are reduced to hydrogen gas.<sup>47</sup> HER competes for active sites and electrons, particularly at the negative potentials required for NO₃RR, thereby lowering the Faradaic efficiency (the percentage of electrons that go to the desired product).<sup>47</sup> The ideal catalyst must therefore not only be active for nitrate reduction but also highly selective for the desired product (e.g., NH₃) while effectively suppressing HER.

#### State-of-the-Art Electrocatalysts

Research efforts have focused on identifying materials that can navigate this complex reaction landscape efficiently.

**Copper-Based Materials**: Copper (Cu) and its oxides (e.g., Cu₂O) have emerged as the most promising class of catalysts for the selective reduction of nitrate to ammonia.<sup>45</sup> Their effectiveness is attributed to the favorable electronic structure of copper; its occupied d-orbitals have energy levels similar to the lowest unoccupied molecular orbital (LUMO) of the nitrate ion, which facilitates the crucial first electron transfer step, often considered the rate-determining step of the reaction.<sup>46</sup> While promising, pure copper materials often require high overpotentials and their activity can be limited.

**Bimetallic and Alloy Catalysts**: To improve upon the performance of pure copper, significant research has focused on bimetallic and alloy catalysts. Introducing a second metal can create powerful synergistic effects by modifying the geometric and electronic structure of the active sites.<sup>51</sup> For example, alloying Cu with noble metals like palladium (Pd) or rhodium (Rh) can enhance the activity for nitrate reduction or alter reaction pathways.<sup>51</sup> Alloying with other elements like bismuth (Bi) or tin (Sn) has been shown to improve selectivity and reduce the formation of unwanted by-products.<sup>51</sup> These studies demonstrate that the design space for NO₃RR catalysts is vast and combinatorial, involving not just the choice of elements but also their ratio, morphology, and surface structure.

#### Unresolved Challenges and the Need for Novel Materials

Despite considerable progress, the performance of current NO₃RR electrocatalysts falls short of the requirements for practical, large-scale application. Key challenges include insufficient activity (low current densities, especially at low nitrate concentrations typical of real wastewater), suboptimal selectivity, and poor long-term operational stability due to catalyst degradation.<sup>46</sup> The vast compositional space of multi-metallic alloys, particularly complex systems like high-entropy alloys (HEAs) containing five or more principal elements, remains largely unexplored through traditional methods. This defines a clear scientific gap: a critical need exists for a high-throughput, systematic, and predictive approach to explore this vast design space and discover new catalyst compositions with superior performance. Table 1 provides a quantitative benchmark of the current state of the art, against which any newly discovered materials must be measured.

#### Table 1: Comparison of State-of-the-Art Electrocatalysts for the Nitrate Reduction Reaction

| Catalyst Composition | Synthesis Method | Electrolyte Conditions | NH₃ Faradaic Efficiency (%) | NH₃ Partial Current Density (mA/cm²) | Key Findings / Limitations |
|---------------------|------------------|----------------------|----------------------------|-----------------------------------|---------------------------|
| Pd incorporated Cu₂O<sup>52</sup> | One-pot solution synthesis | Not Specified | Up to 97.4 | Not Specified | High selectivity achieved; performance metrics depend heavily on Pd loading and morphology. |
| Pd-Cu on Stainless Steel (SS)<sup>51</sup> | Electrodeposition | Not Specified | Not specified for NH₃ | Higher than Cu/SS | Pd addition significantly increases nitrate reduction activity, but favors N₂ production over NH₃. |
| Cu/Cu₂O Nanowire Arrays<sup>45</sup> | Electrochemical reconstruction | 0.1 M K₂SO₄ with 0.1 M KNO₃ | ~95 | ~125 at -0.55 V vs RHE | In situ reconstruction forms active Cu/Cu₂O interface; stability over long-term operation is a concern. |
| Rh clusters on Cu Nanowires (Rh@Cu)<sup>46</sup> | Not Specified | Not Specified | >90 | High (value not specified) | Demonstrates synergistic effect where Rh activates hydrogenation on Cu sites, boosting NH₃ production. |
| Cu-Bi Bimetallic Cathode<sup>51</sup> | Not Specified | Not Specified | Not specified for NH₃ | Improved over Cu/Fe | Bimetallic system improves performance and reduces by-product formation compared to single metals. |

Works cited:
A Comprehensive Survey of Scientific Large Language Models and ..., accessed September 15, 2025, https://www.researchgate.net/publication/386188217_A_Comprehensive_Survey_of_Scientific_Large_Language_Models_and_Their_Applications_in_Scientific_Discovery
A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery - ACL Anthology, accessed September 15, 2025, https://aclanthology.org/2024.emnlp-main.498.pdf
Application of artificial intelligence large language models in drug target discovery - PMC, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12279696/
Editorial – The Use of Large Language Models in Science: Opportunities and Challenges, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10485814/
How should the advancement of large language models affect the practice of science? | PNAS, accessed September 15, 2025, https://www.pnas.org/doi/10.1073/pnas.2401227121
[2406.10833] A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery - arXiv, accessed September 15, 2025, https://arxiv.org/abs/2406.10833
HoneyComb: A Flexible LLM-Based Agent System for Materials Science - ACL Anthology, accessed September 15, 2025, https://aclanthology.org/2024.findings-emnlp.192.pdf
Catalysis | PNNL - Pacific Northwest National Laboratory, accessed September 15, 2025, https://www.pnnl.gov/explainer-articles/catalysis
3.1: Principles of Catalysis – Introductory Biochemistry - Open Oregon Educational Resources, accessed September 15, 2025, https://openoregon.pressbooks.pub/biochemistry/chapter/3-1-basic-principles-of-catalysis-biology-libretexts/
Heterogeneous catalyst discovery using 21st century tools: a tutorial - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2014/ra/c3ra45852k
Heterogeneous catalyst discovery using 21st century tools: a tutorial - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlelanding/2014/ra/c3ra45852k
AI-Empowered Catalyst Discovery: A Survey from Classical Machine Learning Approaches to Large Language Models - arXiv, accessed September 15, 2025, https://arxiv.org/html/2502.13626v1
A Machine Learning and Explainable AI Framework Tailored for ..., accessed September 15, 2025, https://pubs.acs.org/doi/10.1021/acs.jpcc.4c05332
Artificial-intelligence-driven discovery of catalyst genes with application to CO2 activation on semiconductor oxides - PMC, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC8776738/
Catalysts by Design: The Power of Theory - PMC - PubMed Central, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC5849427/
Machine Learning Descriptors for Data‐Driven Catalysis Study - PMC - PubMed Central, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10401178/
High-throughput screening and DFT characterization of bimetallic alloy catalysts for the nitrogen reduction reaction - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp01094b
Machine learning accelerates catalyst discovery - BIFOLD Berlin, accessed September 15, 2025, https://www.bifold.berlin/news-events/news/view/news-detail/machine-learning-accelerates-catalyst-discovery
Toward accelerated discovery of solid catalysts using extrapolative machine learning approach | Chemistry Letters | Oxford Academic, accessed September 15, 2025, https://academic.oup.com/chemlett/article/53/8/upae163/7744865
Are LLMs Ready for Real-World Materials Discovery? - arXiv, accessed September 15, 2025, https://arxiv.org/html/2402.05200v1
Integrating Large Language Models into the Chemistry and Materials Science Laboratory Curricula - ACS Publications, accessed September 15, 2025, https://pubs.acs.org/doi/full/10.1021/acs.chemmater.5c00111
(PDF) Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities - ResearchGate, accessed September 15, 2025, https://www.researchgate.net/publication/383792197_Fine-tuning_large_language_models_for_domain_adaptation_Exploration_of_training_strategies_scaling_model_merging_and_synergistic_capabilities
[2412.09560] Foundational Large Language Models for Materials Research - arXiv, accessed September 15, 2025, https://arxiv.org/abs/2412.09560
(PDF) DARWIN Series: Domain Specific Large Language Models for Natural Science, accessed September 15, 2025, https://www.researchgate.net/publication/373451452_DARWIN_Series_Domain_Specific_Large_Language_Models_for_Natural_Science
[2505.03049] 34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery - arXiv, accessed September 15, 2025, https://arxiv.org/abs/2505.03049
(PDF) 34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery - ResearchGate, accessed September 15, 2025, https://www.researchgate.net/publication/391493027_34_Examples_of_LLM_Applications_in_Materials_Science_and_Chemistry_Towards_Automation_Assistants_Agents_and_Accelerated_Scientific_Discovery
From text to insight: large language models for chemical data extraction - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2025/cs/d4cs00913d
Application of Large Language Models in Chemistry Reaction Data Extraction and Cleaning, accessed September 15, 2025, https://lucyinstitute.nd.edu/wp-content/uploads/2025/02/3627673.3679874.pdf
Fine-tuning large language models for chemical text mining - PMC - PubMed Central, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11234886/
Fine-tuning large language models for chemical text mining - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlelanding/2024/sc/d4sc00924j
Fine-tuning Large Language Models for Chemical Text Mining | Organic Chemistry | ChemRxiv | Cambridge Open Engage, accessed September 15, 2025, https://chemrxiv.org/engage/chemrxiv/article-details/65baa07b9138d2316124f224
14 examples of how LLMs can transform materials science and chemistry: a reflection on a large language model hackathon, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10561547/
Evaluating the performance and robustness of LLMs in materials ..., accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00090d
Assessment of fine-tuned large language models for real-world chemistry and material science applications - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc04401k
LLM4Mat-bench: benchmarking large language models for materials property prediction, accessed September 15, 2025, https://www.researchgate.net/publication/391410530_LLM4Mat-bench_benchmarking_large_language_models_for_materials_property_prediction
Could LLMs help design our next medicines and materials? | MIT News, accessed September 15, 2025, https://news.mit.edu/2025/could-llms-help-design-our-next-medicines-and-materials-0409
survey of generative AI for de novo drug design: new frontiers in molecule and protein ... - Oxford Academic, accessed September 15, 2025, https://academic.oup.com/bib/article/25/4/bbae338/7713723
The Age of Generative Chemistry: AI's Impact on Molecule Design - Quantiphi, accessed September 15, 2025, https://quantiphi.com/blog/the-age-of-generative-chemistry-ais-impact-on-molecule-design/
[2402.05200] Are LLMs Ready for Real-World Materials Discovery? - arXiv, accessed September 15, 2025, https://arxiv.org/abs/2402.05200
arxiv.org, accessed September 15, 2025, https://arxiv.org/html/2508.06691v1
Materials science in the era of large language models: a perspective - RSC Publishing, accessed September 15, 2025, https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00074a
What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks - arXiv, accessed September 15, 2025, https://arxiv.org/html/2305.18365v3
(PDF) Materials science in the era of large language models: a ..., accessed September 15, 2025, https://www.researchgate.net/publication/379155355_Materials_science_in_the_era_of_large_language_models_a_perspective
DARWIN 1.5 : Large Language Models as Materials Science Adapted Learners - arXiv, accessed September 15, 2025, https://arxiv.org/html/2412.11970v2
Electrocatalytic Nitrate and Nitrite Reduction toward Ammonia Using Cu2O Nanocubes: Active Species and Reaction Mechanisms - PMC, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11009949/
Copper-Based Electrocatalysts for Nitrate Reduction to Ammonia, accessed September 15, 2025, https://www.mdpi.com/1996-1944/16/11/4000
Robust catalyst assessment for the electrocatalytic nitrate reduction reaction Devesh K. Pathaka, Rajkumar Janaa, Ruth Belloa, K - ChemRxiv, accessed September 15, 2025, https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66b3bba45101a2ffa86ff014/original/robust-catalyst-assessment-for-the-electrocatalytic-nitrate-reduction-reaction.pdf
Electrocatalytic Nitrate and Nitrite Reduction toward Ammonia Using Cu2O Nanocubes: Active Species and Reaction Mechanisms | Journal of the American Chemical Society, accessed September 15, 2025, https://pubs.acs.org/doi/10.1021/jacs.3c13288
Progress and perspectives in the electroreduction of low-concentration nitrate for wastewater management - PMC - PubMed Central, accessed September 15, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11773467/
Electrocatalytic Nitrate Reduction for Brackish Groundwater Treatment: From Engineering Aspects to Implementation - MDPI, accessed September 15, 2025, https://www.mdpi.com/2076-3417/14/19/8986
Recent research progress of electrocatalytic reduction technology for nitrate wastewater: a review - Queen's University Belfast, accessed September 15, 2025, https://pure.qub.ac.uk/files/422662788/Nitrate_review.pdf
(PDF) Electrocatalytic conversion of nitrate waste into ammonia: a review - ResearchGate, accessed September 15, 2025, https://www.researchgate.net/publication/361879132_Electrocatalytic_conversion_of_nitrate_waste_into_ammonia_a_review
Critical review in electrocatalytic nitrate reduction to ammonia towards a sustainable nitrogen utilization | Request PDF - ResearchGate, accessed September 15, 2025, https://www.researchgate.net/publication/377606084_Critical_review_in_electrocatalytic_nitrate_reduction_to_ammonia_towards_a_sustainable_nitrogen_utilization
Cu-based catalysts for electrocatalytic nitrate reduction to ammonia ..., accessed September 15, 2025, https://pubs.rsc.org/en/content/articlelanding/2024/ey/d4ey00002a
Materials Project - Wikipedia, accessed September 15, 2025, https://en.wikipedia.org/wiki/Materials_Project
Materials Project, accessed September 15, 2025, https://next-gen.materialsproject.org/
Materials Project Documentation: Introduction, accessed September 15, 2025, https://docs.materialsproject.org/
Apps Overview - Materials Project, accessed September 15, 2025, https://next-gen.materialsproject.org/apps
NOMAD - FAIRsharing, accessed September 15, 2025, https://fairsharing.org/2501
NOMAD Metainfo, accessed September 15, 2025, https://nomad-lab.eu/services/metainfo
NOMAD, accessed September 15, 2025, https://nomad-lab.eu/
Catalysis Explorer - Materials Project, accessed September 15, 2025, https://next-gen.materialsproject.org/catalysis
Materials Project Data - Registry of Open Data on AWS, accessed September 15, 2025, https://registry.opendata.aws/materials-project/
Database Versions | Materials Project Documentation, accessed September 15, 2025, https://docs.materialsproject.org/changes/database-versions
Discovering catalysts that overcome scaling limitations with high-throughput screening and machine learning - American Chemical Society, accessed September 15, 2025, https://acs.digitellinc.com/p/s/discovering-catalysts-that-overcome-scaling-limitations-with-high-throughput-screening-and-machine-learning-589890
Combined High-Throughput DFT and ML Screening of Transition Metal Nitrides for Electrochemical CO2 Reduction - ACS Publications, accessed September 15, 2025, https://pubs.acs.org/doi/10.1021/acscatal.3c01249
Calculation Details - Materials Project Documentation, accessed September 15, 2025, https://docs.materialsproject.org/methodology/materials-methodology/calculation-details
How machine learning can accelerate electrocatalysis discovery and optimization - Materials Horizons (RSC Publishing), accessed September 15, 2025, https://pubs.rsc.org/en/content/articlelanding/2023/mh/d2mh01279k