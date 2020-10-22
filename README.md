This repo is for the code written for the Computational Lab at University of
Pisa.
This is a work in progress, the main topic will be the implementation of the
Propp-Wilson for the study of the Ising model.

Keywords: Propp-Wilson, Markov chain, MCMC, Coupling from the past, Ising model

A short description for the scripts:
- `vanilla_mcmc.py`: here I use a simple Monte Carlo method with Markov Chains
  for generating a target distribution over a finite set of states.
- `simple_pw.py`: as the name says, it is a Propp Wilson that runs all the N
  chains, with the same distributions of `vanilla_mcmc`.
- `false_pw.py`: same as `simple_pw`, but this one regenerates all the random
  values. It shouldn't work.
- `monotone_pw.py` implements the sandwiching method with a linear chain.
- `monotone_legth_pw.py`  same as above, but plots the numbers of steps needed
  for convergence.
