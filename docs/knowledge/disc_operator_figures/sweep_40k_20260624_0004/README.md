# Discrete Operator 40k Sweep

Source output: `outputs/disc_operator/sweep_40k_20260624_0004`

## Best By Final MSE

- mse10=0.43549, mse1=0.028582, arm=cont, K=0, U=5, seed=0, codes=None
- mse10=0.43735, mse1=0.028928, arm=cont, K=0, U=5, seed=1, codes=None
- mse10=0.48963, mse1=0.02833, arm=cont, K=0, U=5, seed=2, codes=None
- mse10=0.53553, mse1=0.067392, arm=disc_c, K=32, U=5, seed=0, codes=21
- mse10=0.54291, mse1=0.069293, arm=disc_c, K=16, U=5, seed=1, codes=16
- mse10=0.55847, mse1=0.069426, arm=disc_c, K=16, U=5, seed=0, codes=14
- mse10=0.56644, mse1=0.078046, arm=disc_c, K=16, U=5, seed=2, codes=16
- mse10=0.57092, mse1=0.067281, arm=disc_c, K=32, U=5, seed=1, codes=24
- mse10=0.57526, mse1=0.073309, arm=disc_c, K=32, U=5, seed=2, codes=25
- mse10=0.60626, mse1=0.081492, arm=disc_c, K=8, U=5, seed=1, codes=8

## Mean Final MSE By Arm/K/U

- mean=0.45416, std=0.025094, n=3, arm=cont, K=0, U=5
- mean=0.55594, std=0.0097727, n=3, arm=disc_c, K=16, U=5
- mean=0.56057, std=0.017795, n=3, arm=disc_c, K=32, U=5
- mean=0.61834, std=0.0085784, n=3, arm=disc_c, K=8, U=5
- mean=0.67309, std=0.029621, n=3, arm=disc, K=8, U=5
- mean=0.67888, std=0.072735, n=3, arm=disc, K=16, U=5
- mean=0.68258, std=0.077823, n=3, arm=disc, K=32, U=5
- mean=0.68667, std=0.023517, n=3, arm=disc_c, K=32, U=1
- mean=0.73609, std=0.018897, n=3, arm=cont, K=0, U=1
- mean=0.74813, std=0.035274, n=3, arm=disc_c, K=4, U=5
- mean=0.76156, std=0.0052227, n=3, arm=disc_c, K=16, U=1
- mean=0.76354, std=0.012867, n=3, arm=disc, K=4, U=5
- mean=0.80392, std=0.011799, n=3, arm=disc, K=32, U=1
- mean=0.85351, std=0.019879, n=3, arm=disc, K=16, U=1
- mean=0.8732, std=0.0053881, n=3, arm=disc_c, K=8, U=1
- mean=0.93214, std=0.012314, n=3, arm=disc, K=8, U=1
- mean=0.95346, std=0.0042068, n=3, arm=disc_c, K=4, U=1
- mean=0.98985, std=0.0058976, n=3, arm=disc, K=4, U=1

## Interpretation

Continuous U=5 remains best. Contractive discrete operators help versus unconstrained discrete operators, but do not beat the continuous baseline in this run.
