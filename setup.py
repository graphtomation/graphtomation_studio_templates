from setuptools import setup, find_packages


# Define the package name once
package_name = "graphtomation_studio_templates"

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=package_name,
    version="0.0.1",
    author="Aditya Mishra",
    author_email="aditya.mishra@adimis.in",
    description="A Django app for templating in Graphtomation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    include_package_data=True,
    install_requires=[
        "Django>=3.2",
    ],
    classifiers=[
        "Framework :: Django",
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    license="Proprietary",
)
