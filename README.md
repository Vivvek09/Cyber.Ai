<h1>Cyber.Ai</h1>

This is a **Cybersecurity Platform Featuring an AI-Driven Assessment Bot** which will go through a company's infrastructure, probing for vulnerabitlites and gaps.
Once identified, it would ask dynamic questions to gain further clarifications. The topics include Network Security, Compliance, Data Protection and more.
Based on the answers by the company, it would rate every answer and give recommendations to improve if required. At the end of the assessment, an overall assessment report would be generated
highlighting the weak points as well as suggest actionable insights.

We needed to heavily rely on understanding the **context** of what was present in the documentats as these documents would rarely have the exact terminologies for the protocols followed.

<h4>Key Features</h4>

1. 🧑🏻‍💻 Dynamic and Context-Aware Questioning using a LLM and Chain of Thought Prompt Engineering.
2. 🤔 Contextual Learning with Knowledge Graphs to improve the accuracy of dynamic questioning.
3. 📝 Assessment of Infrastructure Documentation to find vulnerabilties to give recommendations.
4. 💯 Real-Time Risk Scoring categorizing risks into different categories based on the knowledge base.
5. ✅ Detailed Report and Recommendations highlighting major risks and conclusions derived from the question-answering.
6. 📈 Topic Modelling for Incident Reports to aid in analysis and visualisation.

<h4>Constraints</h4>

1. We had to use open-source LLMs or fine-tune small language models (SLMs), due to security and privacy reasons.
2. Less computational power was crucial.

<h4>Our Solution</h4>

![image](https://github.com/user-attachments/assets/aacb7962-3632-4b7f-8891-58640d229087)

- We parsed the input document (containing images, text, and tables) using PyPDF to extract text, then preprocessed it.
- A cybersecurity corpus was created covering 12 domains (network, data protection, cloud security, etc.).
- Words from the corpus found in the document were replaced with `<mask>`, generating a masked input.
- Using the [SecureBERT](https://huggingface.co/ehsanaghaei/SecureBERT) model, we probabilistically predicted values for `<mask>`, ranking the top _k_ terms in order of relevance.
- Least relevant predictions indicated topics insufficiently covered, prompting clarification.
- Clustering embeddings of these terms highlighted coverage gaps, aiding context-aware questioning using Chain of Thought Prompt Engineering.
- Iterating through dynamic questions based on chat history and context, we collected enough data to produce a detailed report on vulnerabilities, risk scores, and recommendations.

<h4>Technology Used</h4>

1. PyPDF
2. SecureBERT
3. Phi-3.5-mini-instruct LLM
4. FastAPI
5. Next.js

<h4>Future Improvements</h4>
We think that a right exit strategy is equally important as to determine when to stop asking the questions and generate the report based on how 'complete' the information gathered is. For this we will be exploring knowledge graphs along with google's page ranking approach to determine the completness of the knowledge graph of the uploaded documentation.
