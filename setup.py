from setuptools import setup, find_packages

setup(
    name='xAImed',
    version='0.1',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
    ],
)