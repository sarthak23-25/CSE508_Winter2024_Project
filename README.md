# Legal Lens Group No. 44

## Team Members:
1. Harsh Patel (MT23056)
2. Sahil More (MT23079)
3. Sarthak Pol (MT23082)
4. Vinayak Katoch (MT23105)
5. Yashwant Rana (MT23107)

## Project Overview:
Legal Lens Group No. 44 aims to revolutionize natural language processing tasks in the legal domain by providing accurate and contextually relevant responses to legal queries. By integrating external knowledge bases and employing advanced NLP techniques, Legal Lens facilitates the digital transformation of legal processes, enhances access to legal information, and assists legal professionals and individuals in navigating complex legal scenarios efficiently.

## Workflow:

1. *User Registration:*
   - Users register with Legal Lens using their email and create a password for their account.

2. *Login:*
   - After registration, users can log in to their account using their credentials.

3. *Main Page:*
   - Upon login, users are redirected to the main page of Legal Lens.

4. *BM25 Working Model:*
   - Users can input their legal query into the search bar.
   - The BM25 algorithm retrieves the top 10 relevant results related to the user's query.
   - Upon selecting "Process," the system generates a knowledge graph representation of the top 10 queries.

5. *Knowledge Graph Analysis:*
   - Cosine similarity is calculated among all pairs of the top 10 results.
   - Based on cosine similarity scores, a new ranking is generated, and the top 3 most relevant results are displayed to the user.

6. *Chatbot Interaction:*
   - Users can interact with specialized chatbots in different legal domains such as murder law, child law, maternity law, etc.
   - Each chatbot is trained on specific datasets related to its legal domain, providing accurate and tailored responses.
   - Users can select a specific chatbot based on their legal query and interact with it to receive guidance and information.

7. *Feedback Mechanism:*
   - Users have the option to provide feedback through a dedicated form if they are dissatisfied with the provided information or have suggestions for improvement.
   - The feedback loop helps in continuously refining and improving the system based on user input.

## Functionalities:

1. *Knowledge Graph:*
   - Represents legal concepts, entities, relationships, and rules in a structured format.
   - Organizes legal information into a graph-based system for enhanced understanding and relationships.

2. *BM25 Algorithm:*
   - Retrieves top 10 relevant results based on user queries, enhancing search accuracy in legal documents.

3. *Chatbot System:*
   - Utilizes specialized chatbots trained in different legal domains to provide tailored advice and information to users.
   - Enhances user experience by allowing natural language interactions for legal queries.

4. *Datasets Used:*
   - Indian Penal Code PDFs, Crime PDFs, A1860-45 PDF, 33 Law of Crime, and other legal documents.
   - Government sources such as National Judicial Data Grid (NJDG) & IndianKanoon are utilized for dataset collection and validation.

## References:
1. [Corpus for Automatic Structuring of Legal Documents](https://arxiv.org/abs/2201.13125) - P. Kalamkar et al.
2. [Dependency Learning for Legal Judgment Prediction](https://arxiv.org/abs/2112.06370) - Y. Huang et al.
3. [Named Entity Recognition in Indian court judgments](https://arxiv.org/abs/2211.03442) - P. Kalamkar et al.
4. [SemEval 2023 Task 6: LegalEval - Understanding Legal Texts](https://arxiv.org/abs/2304.09548) - A. Modi et al.
[![Legal Lens Demo](https://i.postimg.cc/7b8xZV9H/Untitled-design.gif)](https://postimg.cc/9wb3xysK)
