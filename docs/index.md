# Introduction

This repo is an implementation of project DC Plus at Elia Group.

DCPlus is based on the paper "[Voltage-sensitive distribution factors for contingency analysis and topology optimization](https://arxiv.org/pdf/2509.19976)". DCPlus is a linearization around the N-0 AC loadflow and approximates the changes in the network by computing an equivalent to one AC Newton-Raphson iteration step. DCPlus is expected to solve about one million loadflows per second on a 2.000 bus network.

Please check out our [full documentation](https://eliagroup.github.io/DCPlus).


## Finding help

If you require help with using this package, your first point of contact is <a href="mailto:loadflowsolver@eliagroup.eu">loadflowsolver@eliagroup.eu</a>.

## Contributing

If you want to contribute to DCPlus, check out our [Contribution Guide](./contribution_guide.md).

## Roadmap

The road map will include a numpy and a jax implementation. The numpy version will be for readability only where the jax version is a gpu optimized implementation.  

- Q1: Core implementation in numpy & jax
    - basic bussplit 
    - N-1 analysis
    - missing: PV to PQ switch (Generator outage)
    -> finally GPU profiling
- Q2: 
    - optimize batched topologies on gpu
    - support multi-outages
    - support full action sets, e.g. with disconnections
    - support shunt outages and reassignments 
