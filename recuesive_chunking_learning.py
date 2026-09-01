"""In this module lets explore how recursive chunking works
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


sample_text = """Python is a popular programming language.

Python is easy to learn and widely used for web development, data science, automation, and artificial intelligence. It has a simple syntax, which makes it beginner-friendly. Python also provides many useful libraries that help developers build applications quickly.

Learning Python step by step can help beginners understand programming concepts such as variables, loops, functions, and classes."""


seperators = ["\n\n", ". "]

splitter = RecursiveCharacterTextSplitter(
    separators=seperators,
    chunk_size=50,
    chunk_overlap=10,
    keep_separator=False
)

chunks = splitter.split_text(sample_text)
for index, chunk in enumerate(chunks):
    print(f"Chunk {index}: length {len(chunk)}\n\n{chunk}\n\n\n")