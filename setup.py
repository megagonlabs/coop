from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="coop",
    version="0.0.1",
    description="Convex Aggregation for Opinion Summarization (Iso et al; Findings of EMNLP 2021)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hayate Iso",
    author_email="hayate@megagon.ai",
    url="https://github.com/megagonlabs/coop",
    packages=find_packages(),
    license="BSD",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.7',
)
