import os 



from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_tavily  import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


# memory = MemorySaver()



embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
      task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

os.makedirs("data/test_files", exist_ok=True)

sample_texts={
    "data/test_files/sample.txt": """
    
    Python: A Comprehensive Overview
Introduction

Python is a high-level, interpreted, general-purpose programming language known for its simplicity, readability, and versatility. It was created by Guido van Rossum and first released in 1991. Python emphasizes clean, easy-to-understand syntax, making it one of the most popular programming languages for beginners and experienced developers alike.

Today, Python is used by millions of developers worldwide for web development, data science, artificial intelligence, machine learning, automation, cybersecurity, cloud computing, game development, desktop applications, and scientific research.

History of Python

Python was developed by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands during the late 1980s. The language was designed to overcome the limitations of the ABC programming language while maintaining simplicity and readability.

Major milestones include:

1991: Python 0.9.0 released
2000: Python 2.0 introduced garbage collection and Unicode support
2008: Python 3.0 released with significant improvements and backward-incompatible changes
2020: Official support for Python 2 ended
Today: Python 3 continues to evolve with regular performance improvements and new language features.
Features of Python

Python provides numerous features that make it one of the most widely used programming languages.

1. Easy to Learn

Python's syntax closely resembles natural English, making it easy for beginners to understand and write programs.

Example:

name = "Manoj"
print(f"Hello {name}")
2. High-Level Language

Python abstracts low-level details such as memory management, allowing developers to focus on solving business problems rather than hardware-level operations.

3. Interpreted Language

Python programs are executed line by line by the Python interpreter instead of being compiled directly into machine code.

Advantages include:

Faster development
Easier debugging
Platform independence
4. Object-Oriented Programming

Python supports object-oriented programming (OOP), enabling developers to organize code using classes and objects.

Example:

class Student:
    def __init__(self, name):
        self.name = name

    def introduce(self):
        return f"My name is {self.name}"

student = Student("Manoj")
print(student.introduce())
5. Dynamically Typed

Variables do not require explicit type declarations.

age = 25
name = "Python"
salary = 45000.75

Python determines the data type automatically.

6. Cross-Platform

Python programs run on:

Windows
Linux
macOS
Raspberry Pi
Cloud platforms

with little or no modification.

7. Extensive Standard Library

Python includes a rich collection of built-in modules such as:

os
sys
math
datetime
pathlib
json
sqlite3
threading
multiprocessing
asyncio

This extensive standard library is often referred to as Python's "batteries included" philosophy.

8. Large Ecosystem

Python has hundreds of thousands of third-party libraries available through the Python Package Index (PyPI).

Popular examples include:

NumPy
Pandas
Matplotlib
TensorFlow
PyTorch
Scikit-learn
FastAPI
Flask
Django
SQLAlchemy
LangChain
Applications of Python

Python is used across numerous domains.

Web Development

Frameworks:

Django
Flask
FastAPI

Applications:

REST APIs
E-commerce websites
Social media platforms
SaaS applications
Data Science

Python is widely used for:

Data analysis
Data cleaning
Visualization
Statistical analysis

Libraries:

Pandas
NumPy
Matplotlib
Seaborn
Plotly
Artificial Intelligence

Python dominates AI development.

Applications include:

Chatbots
Recommendation systems
Large Language Models (LLMs)
Natural Language Processing
Computer Vision

Popular libraries:

TensorFlow
PyTorch
Hugging Face Transformers
LangChain
LangGraph
Machine Learning

Python supports the complete machine learning lifecycle:

Data preprocessing
Model training
Model evaluation
Deployment
Monitoring
Automation

Python automates repetitive tasks such as:

File management
Email automation
Excel processing
PDF generation
Web scraping
Browser automation

Libraries:

Selenium
BeautifulSoup
Playwright
Requests
Cloud Computing

Python is widely used with cloud platforms like AWS, Azure, and Google Cloud.

Common tasks:

Serverless functions
Cloud automation
Infrastructure management
API development
Cybersecurity

Python is extensively used for:

Ethical hacking
Network scanning
Packet analysis
Malware analysis
Penetration testing

Libraries:

Scapy
Paramiko
Pwntools
Internet of Things (IoT)

Python powers embedded systems and IoT applications using platforms such as Raspberry Pi.

Desktop Applications

Frameworks:

Tkinter
PyQt
Kivy
PySide
Python Data Types

Python includes several built-in data types.

Numeric:

int
float
complex

Text:

str

Boolean:

bool

Collections:

list
tuple
set
dict

Binary:

bytes
bytearray
memoryview
Programming Paradigms

Python supports multiple programming paradigms:

Procedural Programming
Object-Oriented Programming
Functional Programming
Modular Programming
Event-Driven Programming
Asynchronous Programming

This flexibility allows developers to choose the most suitable style for a given problem.

Advantages of Python
Easy to read and write
Beginner-friendly syntax
Large and active community
Massive collection of libraries
Cross-platform compatibility
Rapid application development
Strong support for AI and machine learning
Open source and free to use
Excellent documentation
High developer productivity
Limitations of Python
Slower execution compared to compiled languages like C++ or Rust
Higher memory consumption
Global Interpreter Lock (GIL) can limit CPU-bound multithreading
Less suitable for low-level system programming
Not ideal for performance-critical applications such as game engines or operating systems

Despite these limitations, Python remains an excellent choice for most modern software development tasks.

Python in Modern Software Development

Python has become a foundational language in many industries.

It is commonly used to build:

RESTful APIs
AI-powered applications
Retrieval-Augmented Generation (RAG) systems
Multi-agent AI systems
Data engineering pipelines
Web applications
Automation scripts
DevOps tools
Cloud-native services
Scientific computing applications

Its integration with modern frameworks such as FastAPI, LangChain, LangGraph, and various AI ecosystems has made Python one of the primary languages for developing intelligent software systems.

Conclusion

Python is one of the most influential and versatile programming languages in the world. Its clear syntax, extensive standard library, thriving ecosystem, and support for multiple programming paradigms make it suitable for beginners, researchers, startups, and large enterprises alike. Whether developing web applications, automating workflows, analyzing data, or building advanced AI systems, Python provides the tools and flexibility needed to create robust, scalable, and maintainable software. As technologies such as artificial intelligence, cloud computing, and data science continue to expand, Python is expected to remain a leading programming language for years to come.
    
    
    
    """,
    
    "data/test_files/sample2.txt": """ 
    
    # Artificial Intelligence (AI) and Machine Learning (ML): A Comprehensive Overview

# Introduction

Artificial Intelligence (AI) is a branch of computer science that focuses on creating intelligent systems capable of performing tasks that typically require human intelligence. These tasks include reasoning, learning, problem-solving, perception, language understanding, decision-making, and creativity. AI enables machines to analyze information, adapt to new situations, and make decisions with minimal human intervention.

Machine Learning (ML) is a subset of Artificial Intelligence that allows computers to learn from data instead of being explicitly programmed. ML algorithms identify patterns, build predictive models, and improve their performance over time through experience. Rather than following fixed instructions, machine learning systems continuously refine their predictions as they process more data.

Today, AI and ML power many technologies that people use daily, including search engines, recommendation systems, voice assistants, autonomous vehicles, fraud detection systems, medical diagnosis tools, chatbots, and generative AI applications such as ChatGPT.

---

# Evolution of Artificial Intelligence

The concept of intelligent machines has existed for decades, but AI became an active field of research in 1956 during the Dartmouth Conference, where the term "Artificial Intelligence" was officially introduced.

Important milestones include:

* **1956:** Birth of Artificial Intelligence as a research field.
* **1980s:** Rise of expert systems for decision-making.
* **1997:** IBM Deep Blue defeated world chess champion Garry Kasparov.
* **2012:** Deep learning revolution driven by advances in neural networks and GPUs.
* **2016:** AlphaGo defeated Go champion Lee Sedol.
* **2022–Present:** Rapid growth of Generative AI and Large Language Models (LLMs) such as GPT, Gemini, Claude, Llama, and Qwen.

---

# What is Artificial Intelligence?

Artificial Intelligence is the science and engineering of developing computer systems that can simulate human intelligence. AI systems perceive their environment, process information, learn from experience, and make intelligent decisions to achieve specific goals.

AI combines multiple disciplines, including:

* Computer Science
* Mathematics
* Statistics
* Cognitive Science
* Linguistics
* Robotics
* Data Science

---

# Types of Artificial Intelligence

## 1. Narrow AI (Weak AI)

Narrow AI is designed to perform a specific task efficiently. It cannot operate beyond its programmed domain.

Examples include:

* Voice assistants
* Recommendation systems
* Email spam filters
* Face recognition
* Language translation

Most AI applications available today belong to this category.

---

## 2. General AI (Strong AI)

General AI refers to systems capable of understanding, learning, and performing any intellectual task that a human can perform. Such systems would possess reasoning, planning, creativity, and common sense across a wide variety of domains.

General AI remains a long-term research goal and has not yet been achieved.

---

## 3. Super AI

Super AI is a hypothetical stage in which artificial intelligence surpasses human intelligence in every field, including scientific discovery, creativity, emotional understanding, and strategic decision-making. It remains theoretical and raises important ethical and safety considerations.

---

# What is Machine Learning?

Machine Learning is a subset of AI that focuses on developing algorithms capable of learning patterns from data and making predictions or decisions without explicit programming.

The general workflow of a machine learning system includes:

1. Collect data.
2. Clean and preprocess the data.
3. Select appropriate features.
4. Train a machine learning model.
5. Evaluate its performance.
6. Deploy the model.
7. Continuously monitor and improve it with new data.

---

# Types of Machine Learning

## Supervised Learning

In supervised learning, models are trained using labeled datasets, where the correct output is known.

Common algorithms:

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machines
* Neural Networks

Applications:

* House price prediction
* Spam email detection
* Medical diagnosis
* Credit risk assessment

---

## Unsupervised Learning

Unsupervised learning discovers hidden structures or relationships within unlabeled data.

Common algorithms:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN
* Principal Component Analysis (PCA)

Applications:

* Customer segmentation
* Market basket analysis
* Fraud detection
* Anomaly detection

---

## Semi-Supervised Learning

Semi-supervised learning combines a small amount of labeled data with a larger amount of unlabeled data to improve learning performance.

Applications include image classification, speech recognition, and medical imaging.

---

## Reinforcement Learning

In reinforcement learning, an agent interacts with an environment and learns through rewards and penalties to maximize long-term success.

Applications include:

* Robotics
* Autonomous vehicles
* Game-playing AI
* Resource optimization

---

# Deep Learning

Deep Learning is a specialized branch of Machine Learning that uses artificial neural networks with multiple layers to learn complex patterns directly from large datasets.

Popular architectures include:

* Artificial Neural Networks (ANN)
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Long Short-Term Memory (LSTM)
* Transformers

Deep learning powers many modern AI systems, including image recognition, speech recognition, machine translation, and large language models.

---

# Generative AI

Generative AI focuses on creating new content such as text, images, audio, video, software code, and 3D objects.

Examples include:

* ChatGPT
* Gemini
* Claude
* Llama
* Qwen
* Stable Diffusion
* Midjourney

Applications:

* Content generation
* Programming assistance
* Image generation
* Video creation
* Personalized tutoring
* Customer support

---

# Core Components of AI Systems

Modern AI systems typically consist of:

* Data Collection
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Model Deployment
* Continuous Monitoring
* Model Retraining

High-quality data and continuous improvement are essential for maintaining reliable AI systems.

---

# Applications of AI and ML

Artificial Intelligence and Machine Learning have transformed many industries.

## Healthcare

* Disease diagnosis
* Medical imaging analysis
* Drug discovery
* Personalized treatment
* Patient monitoring

## Finance

* Fraud detection
* Credit scoring
* Algorithmic trading
* Risk assessment
* Financial forecasting

## E-commerce

* Product recommendations
* Dynamic pricing
* Inventory optimization
* Customer analytics
* Personalized marketing

## Manufacturing

* Predictive maintenance
* Quality inspection
* Industrial automation
* Supply chain optimization

## Transportation

* Self-driving vehicles
* Route optimization
* Traffic prediction
* Fleet management

## Education

* Intelligent tutoring systems
* Automated grading
* Personalized learning
* Language learning assistants

## Agriculture

* Crop disease detection
* Precision farming
* Weather prediction
* Yield estimation

## Cybersecurity

* Threat detection
* Malware analysis
* Intrusion detection
* Security automation

---

# Advantages of AI and ML

* Automates repetitive tasks
* Processes massive datasets efficiently
* Improves decision-making
* Reduces operational costs
* Increases productivity
* Provides personalized user experiences
* Enables predictive analytics
* Operates continuously without fatigue
* Supports innovation across industries

---

# Challenges and Limitations

Despite their benefits, AI and ML face several challenges:

* Data privacy concerns
* Algorithmic bias
* High computational requirements
* Dependence on large datasets
* Lack of explainability in complex models
* Security vulnerabilities
* Ethical and legal considerations
* Environmental impact of training large models

Responsible AI development requires fairness, transparency, accountability, and human oversight.

---

# Future of AI and ML

The future of AI and ML includes significant advances in:

* Agentic AI systems
* Autonomous robotics
* Multimodal AI
* Explainable AI (XAI)
* Edge AI
* AI-powered scientific research
* Personalized healthcare
* Smart cities
* Human-AI collaboration

As these technologies continue to mature, they are expected to reshape industries, create new opportunities, and change the way people work and interact with technology.

---

# Conclusion

Artificial Intelligence and Machine Learning are among the most transformative technologies of the 21st century. AI enables machines to perform tasks that traditionally required human intelligence, while Machine Learning allows systems to improve through experience by learning from data. Together, they power applications ranging from healthcare and finance to education, transportation, and generative AI. As research and innovation continue, AI and ML will play an increasingly important role in solving complex global challenges, enhancing productivity, and creating intelligent systems that improve everyday life. Understanding their principles, applications, and ethical implications is essential for students, researchers, and professionals preparing for the future of technology.

    
    """
}






# loading a single text files



for file_path,content in sample_texts.items():
    with open(file_path, "w+", encoding="utf-8") as file_cursor:
        file_cursor.write(content)
        
        

dir_loader = DirectoryLoader("data/test_files",glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"})

documents = dir_loader.load()

# Method 1 : Character Text Splitter
text = documents[0].page_content

print("CHARACTER TEXT SPLITTER")




# char_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=200,
#     chunk_overlap=20,
#     length_function=len
# )

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n"," ",""],
    length_function=len,
)

chunks = char_splitter.split_documents(documents)

chunk_embeddings = embeddings.embed_documents(
    [doc.page_content for doc in chunks]
)

print("Total Chunks:", len(chunks))
print("Embedding Dimension:", len(chunk_embeddings[0]))

# print("Created character chunks:", len(char_chunks))
# print("First chunk length:", len(char_chunks[0]))
# print("First chunk:", char_chunks[0])
# print("Last chunk:", char_chunks[-1])
# print("Chunks combined:", "".join(char_chunks))



# for index,document in enumerate(documents):
#     print(f"Document {index+1}:")
#     print(f"Content: {document.page_content} characters")
#     print(f"Metadata: {document.metadata["source"]} ")



# loader  = TextLoader("data/test_files/sample.txt",encoding="utf-8")

# documents = loader.load()
# print(type(documents))
# print(documents)

# doc=Document(page_content="This is the main text content that will be embedded and searched",metadata={
#     "source": "example.pdf",
#     "author": "Manoj Santra",
#     "page":1,
#     "title": "Example Document 1",
#     "creation_date": "2023-01-01",
#     "keywords": ["example", "document", "search"]
# })

# print("document structure")
# print(f"content : {doc.page_content}")
# print(f"metadata : {doc.metadata}")