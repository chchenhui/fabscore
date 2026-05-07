from setuptools import setup, find_packages

# Core dependencies
required_packages = [
    "aiofiles>=24.1.0",
    "asyncpg>=0.30.0",
    "dataclasses_json>=0.6.7",
    "fastapi>=0.115.12",
    "grpcio>=1.70.0",
    "httpx>=0.28.1",
    "loguru>=0.6.0",
    "matplotlib>=3.10.3",
    "networkx>=3.4.2",
    "numpy>=2.2.5",
    "pandas>=2.2.3",
    "protobuf>=6.30.2",
    "pydantic>=2.11.4",
    "pyvis>=0.3.2",
    "seaborn>=0.13.2",
    "setuptools>=75.6.0",
    "tqdm>=4.67.1",
    "uvicorn[standard]>=0.34.2",
    "faiss-cpu>=1.9.0",
    "pyyaml",
    "openai",
    "volcengine-python-sdk",
    "scipy>=1.7.0",
    "statsmodels>=0.13.0",
    "pymc>=4.0.0",
    "pingouin>=0.5.0",
    "pymupdf4llm>=0.0.16",
    "requests",
]

# Optional dependencies for tuning
tune_extras = [
    "mlflow",
    "torch",
    "peft>=0.15.1",
    "transformers",
    "trl>=0.16.0",
    "vllm",
]

setup(
    name='YuLan-OneSim',
    version='1.0.0',
    author='Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen',
    author_email='wanglei154@ruc.edu.cn',
    description='YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RUC-GSAI/YuLan-OneSim',
    project_urls={
        'Paper': 'https://arxiv.org/abs/2505.07581',
    },
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=required_packages,
    extras_require={'tune': tune_extras},
    python_requires='>=3.10,<3.11',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Sociology',
    ],
    entry_points={
        'console_scripts': [
            'yulan-onesim-cli=main:cli_entry_point',
            'yulan-onesim-server=app:start_server',
        ],
    },
)
