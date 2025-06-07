from setuptools import setup, find_packages

setup(
    name='simple-agent',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple agent using the autogen framework',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'autogen-framework',  # Replace with the actual package name if different
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)