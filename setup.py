from setuptools import setup, find_packages

setup(
    name="mm_203_python_pro",                
    version="0.1",                   
    packages=find_packages(where="src"),  
    package_dir={"": "src"},          
    install_requires=[                
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "yfinance",
        "pybacktestchain",
        
    ],
    include_package_data=True,        
    description="Backtest portfolio strategy",
    author="Melissa Mesnard",
    author_email="melissa.mesnard@dauphine.eu",
    classifiers=[                     
        "Programming Language :: Python :: 3.12.7",
    ],
)
