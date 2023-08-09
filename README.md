# GateON

[![Contributors]][contributors-url]
[![Forks]][forks-url]
[![Issues]][issues-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/martinbarry59/GateON">
    <img src="https://i0.wp.com/procsim.ch/wp-content/uploads/2019/12/1280px-Logo_EPFL.svg_.png?ssl=1" alt="Logo">
  </a>

  <h3 align="center">GateON</h3>

  <p align="center">
    <a href="https://github.com/martinbarry59/GateON/issues">Report Bug</a>
    ·
    <a href="https://github.com/martinbarry59/GateON/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#code">Code</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

GateON is a continual learning algorithm based on context gating of activity and learning rate modulation based on a leaky availability of neurons or parameters [Fast adaptive learning in volatile environment](https://doi.org/10.1101/2022.09.13.507727) 
### Built With

SpikesumNet is a pytorch implemented neural network. few other libraries are used for plotting mainly

* [Pytorch](https://pytorch.org/)
* [matplotlib](https://matplotlib.org/)
* [tqdm](https://tqdm.github.io/)
* [jupyter](https://jupyter.org/)


<!-- GETTING STARTED -->
## Getting Started

Installation of the does not require special software but for python. and python dependencies

### Prerequisites

* python
  to instal python follow the [installation guide](https://realpython.com/installing-python/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/martinbarry59/GateON.git
   ```
2. Install libraries (go to the repository folder)
   ```sh
   pip install -r requirements.txt
   ```
<!-- USAGE EXAMPLES -->

<!-- ROADMAP -->
## Roadmap

See the [open issues](hhttps://github.com/martinbarry59/GateON/issues) for a list of proposed features (and known issues).

## Code

* To run the code:

```
python run_context_network
```

(to find the arguments you can change you can have a look to the function [get_option](https://github.com/martinbarry59/GateON/blob/main/pkg/general_utils.py)).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Swiss National Science foundation](http://www.snf.ch/en/Pages/default.aspx)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-url]: https://github.com/martinbarry59/GateON/graphs/contributors
[forks-url]: https://github.com/martinbarry59/GateON/network/members
[issues-url]: https://github.com/martinbarry59/GateON/issues
