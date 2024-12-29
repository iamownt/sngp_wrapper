from setuptools import setup, find_packages

setup(
    name='sngp_wrapper',
    version='0.1.0',  # Update this to your desired version
    packages=find_packages(),
    description='A wrapper for SNGP functionality',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tao Wang',  # Replace with your name
    author_email='seatao.wang@connect.polyu.hk',  # Replace with your email
    url='https://github.com/iamownt/sngp_wrapper',  # Update with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Update as per your package requirements
)
