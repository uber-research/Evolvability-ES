# Evolvability ES
This repository contains source code for the evolvability-optimizing algorithm evolvability ES.
There are two variants of evolvability ES, MaxVar and MaxEnt; also implemented is standard ES for comparison.

Simple runner scripts are provided for each of the three algorithms: standard.py, maxvar.py, and maxent.py.
Two environments are implemented, and may be switched between with the --env-name flag (accepting values `cheetah` and `ant`).

The Dask Distributed framework is used for communication between different compute nodes.
The runner scripts currently use a local Dask cluster--this could be modified to run on an actual cluster for larger-scale experiments.

We also provide code to run evolvability ES on the simple interference pattern task, given in interference.py.

Dependencies are listed in requirements.txt.
