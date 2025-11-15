from setuptools import setup, find_packages

setup(
    name="BrandonCFBRatings",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "requests",
        "matplotlib",
        "streamlit",
        "plotly",
        "python-dotenv",
    ],
)