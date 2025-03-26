from setuptools import setup, find_packages

setup(
    name="MediBot App",
    version="0.1",
    author="Vinay",
    author_email="vinay@gmail.com",
    packages=find_packages(),
    install_requires=["sentence-transformers==2.2.2",
                      "langchain",
                      "flask",
                      "flask-cors",
                      "pypdf",
                      "python-dotenv",
                      ]
                      )