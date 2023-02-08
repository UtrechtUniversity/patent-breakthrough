# patent-breakthrough

![tests](https://github.com/UtrechtUniversity/patent-breakthrough/actions/workflows/python-package.yml/badge.svg)

Package for the RE Patent Breakthrough project

The code in this repository is used to identify breakthrough innovations in historical patents from the [USPTO](https://www.uspto.gov/).
The docembeddings package contains a variety of methods for creating document embeddings. For this project these methods are optimized and tested for their ability to predict similarity between patents. This is done by optimizing the cosine similarity between patents that are classified into the same technology class. The best performing methods are then used to create document embeddings that are then used to create novelty scores using cosine similarities between the focal patent and patents in previous n years and subsequent n years.

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Project Title](#patent-breakthrough)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
  - [About the Project](#about-the-project)
    - [Built with](#built-with)
    - [License](#license)
    - [Attribution and academic use](#attribution-and-academic-use)
  - [Contributing](#contributing)
  - [Contact](#contact)

<!-- GETTING STARTED -->
## Getting Started

Clone this repository to your working station

```
git clone https://github.com/UtrechtUniversity/patent-breakthrough.git
```

### Prerequisites

To install and run this project you need to have the following prerequisites installed.
```
- Python [>=3.8, <3.11]
- jupyterlab (or any other program to run jupyter notebooks)
```
To install jupyterlab:
```
pip install jupyterlab
```

### Installation

To run the project, ensure to install the project's dependencies.

```sh
cd patent-breakthrough
pip install .
```

### Usage
Open `main.ipynb` using jupyterlab

[TO BE EXPANDED]

<!-- ABOUT THE PROJECT -->
## About the Project

**Date**: February 2023

**Researcher(s)**:

- Benjamin Cornejo Costas (b.j.cornejocostas@uu.nl)

**Research Software Engineer(s)**:

- Raoul Schram
- Shiva Nadi
- Maarten Schermer
- Casper Kaandorp
- Jelle Treep (h.j.treep@uu.nl)

### Built with

This section should list any major frameworks used to build this project.

- [scikit-learn](https://scikit-learn.org/)
- [gensim](https://pypi.org/project/gensim/)
- [sbert](https://www.sbert.net/)

<!-- Do not forget to also include the license in a separate file(LICENSE[.txt/.md]) and link it properly. -->
### License

The code in this project is released under [MIT license](LICENSE).

### Attribution and academic use

Manuscript in preparation

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- NOTES -->
<!-- CONTACT -->
## Contact

Benjamin Cornejo Costas - b.j.cornejocostas@uu.nl

Project Link: [https://github.com/UtrechtUniversity/patent-breakthrough](https://github.com/UtrechtUniversity/patent-breakthrough)
