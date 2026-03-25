<!-- markdown-link-check-disable -->

[![CI](https://github.com/eliagroup/DCPlus/actions/workflows/ci.yaml/badge.svg)](https://github.com/eliagroup/DCPlus/actions/workflows/ci.yaml)

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
<!-- markdown-link-check-enable -->

# DCPlus

-------
## About The Project

This repo builds DCPlus loadflow solver implementation from Elia Group. This DCPlus loadflow solver is based on the paper "[Voltage-sensitive distribution factors for contingency analysis and topology optimization](https://arxiv.org/pdf/2509.19976)". It is a linearization around the N-0 AC loadflow and approximates the changes in the network by computing an equivalent to one AC Newton-Raphson iteration step.

## Getting Started

If you want to get started with the engine, we recommend the test [test_lodf_jax_full_rank_update_compare_powsybl](./tests/jax/test_lodf_jax.py) to get an idea of how to use the solver. A more user-friendly introductory notebook will be added to the `notebooks/` directory at a later point.  


### Prerequisites

If you want to contribute to this repository, we recommend using VS Code's Devcontainer Environment. This allows the developers to use the same environment to develop in.

For this setup, you need to install:
1. `uv`
2. `Microsoft VS Code`
3. `Docker`

### Installation

You can follow our installation guide on our [Contributing page](./CONTRIBUTING.md#local-development-setup).

# Usage

In order to understand the functionalities of this repo, please have a look at our examples in `notebooks/`.
There you can find several Jupyter notebooks that explain how to use the engine.
For example, you can load a grid file and compute the DC loadflow using our GPU-based loadflow solver.
Or you can load an example grid and minimise the branch overload by running the topology optimizer.

You can also build the documentation and open it on your web browser by running
```bash
uv sync --all-groups
uv run mkdocs serve
```

---

## Contributing

Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Distributed under MPL 2.0. See [LICENSE](./LICENSE).

## Citation

If you use our work in scientific research, please cite [our paper on loadflowsolving](https://arxiv.org/abs/2501.17529) and soon also the work on the optimizer architecture, which is to be released soon.

---

## Contact



Team – [loadflowsolver@eliagroup.eu](mailto:loadflowsolver@eliagroup.eu)

---

## Acknowledgments
Thanks to the Energy Transition Fund of the Federal Public Service Economy for their support of the OptOmni project, which led to the development of DC+.| [SPF Economie](https://economie.fgov.be/fr)
 

We credit the authors of JAX.
```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

-----
-----
