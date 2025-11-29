"""
Setup script for the Fine-FinTech project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]
else:
    requirements = []

setup(
    name="fine-fintech",
    version="1.0.0",
    description="Fine-tuning Gemma 3 for Financial Q&A with Enhanced Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fine-FinTech Team",
    author_email="contact@fine-fintech.com",
    url="https://github.com/your-username/Fine-FinTech",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fine-fintech-setup=scripts.setup_environment:main",
            "fine-fintech-train=scripts.train:main",
            "fine-fintech-evaluate=scripts.evaluate:main",
            "fine-fintech-deploy=scripts.deploy:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "deep-learning",
        "transformers",
        "gemma",
        "financial-ai",
        "question-answering",
        "fine-tuning",
        "reasoning",
        "finance",
        "nlp"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/Fine-FinTech/issues",
        "Source": "https://github.com/your-username/Fine-FinTech",
        "Documentation": "https://github.com/your-username/Fine-FinTech#readme",
    },
)