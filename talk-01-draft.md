# Cover


# Fluid dynamics simulation can be hard
General challenging
- non-linear
- chaotic (small perturbation, large difference)
- expensive computation

LBM
- non-linear (converted non-linear terms of navior-stolks into f_equilibrium in cells running in parallel)
- chaotic (small perturbation, large difference): use finer grid to catch it (compensate computing time via parallelization)
- expensive computation (particularly parallelization naturally)

# learn the non-linear
the non-linear of LBM is essentially the particle probability density products.

<show the binary collision term of Boltzmann Transport equation>

<show the loss function from Corbetta (2023)>

# TG simulation of models - what is Tayler-Green

# TG simulation of models - results
TG, mathmatically, closed-form

TG simulation with naive model
<show the TG figure of naive model: place folder of png>

TG simulation with GAVG model
<show the TG figure of GAVG model: place folder of png>

# Karman Vortex street simulation
more complex simulation scenario. commonly seen in nature

<show the karman vortex stree simulation gif: placeholder>

<show the karman vortex stree realworld video gif such as wind tunnel, wind passing bridge, chinney. horizontal gif to fit the slide protrait: placeholder>

## beyond GAVG - ResNet

<placeholder: result of resnet>

why resnet is better?

why LENN is better?

# Future work
- LENNs
- learning more operators
- real world simulations (medical, supernova hydrodynamics etc.)
- domain boundary with surrogate models

# Future work - more
- more layers for naive models

# Appendix
## Corbetta, Gabbana et al., Eur. Phys. J. E (2023) 
## Slack workspace - how this project was cooked
## How to use genAD
