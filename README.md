This repo is for the code written for the Computational Lab at University of
Pisa.
This is a work in progress, the main topic will be the implementation of the
Propp-Wilson algorithm for the study of the Ising model.

Keywords: Propp-Wilson, Markov chain, MCMC, Coupling from the past, Ising model

A short description of the scripts:
- `vanilla_mcmc.py`: here I use a simple Monte Carlo method with Markov Chains
  for generating a target distribution over a finite set of states.
- `simple_pw.py`: as the name says, it is a Propp Wilson that runs all the N
  chains, with the same distributions of `vanilla_mcmc`.
- `false_pw.py`: same as `simple_pw`, but this one regenerates all the random
  values. It shouldn't work.
- `monotone_pw.py`: implements the sandwiching method with a linear chain.
- `monotone_legth_pw.py`:  same as above, but plots the numbers of steps needed
  for convergence.
- `simple_ising.py`: it is a simple ising model, with the pw algorithm. Prints
  the log of the number of steps needed for coalescence and the ratio between
  the number of +1 and the total number of nodes.
