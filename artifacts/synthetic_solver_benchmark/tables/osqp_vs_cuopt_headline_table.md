| Problem family | Largest common solved size | OSQP median end-to-end time at that size | cuOpt median end-to-end time at that size | End-to-end speedup (OSQP/cuOpt) | Solve-only speedup (OSQP/cuOpt) | Faster solver at n=100 | Faster solver at the largest common solved size | First measured crossover size | OSQP strict-optimal count | cuOpt strict-optimal count | Timeout/failure count | Maximum canonical objective gap | Maximum original primal violation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LP-RANGED | 3000 | 1.06274 | 0.385989 | 2.75329 | 2.7882 | OSQP | cuOpt | 3000 | 45 | 45 | 0 | 3.2576e-05 | 1.74157e-07 |
| LP-MIXED | 10000 | 123.403 | 4.18507 | 29.4865 | 0.88649 | OSQP | cuOpt | 3000 | 45 | 45 | 0 | 1.50362e-10 | 2.98748e-10 |
| QP-DIAGONAL | 10000 | 5.78639 | 7.15392 | 0.808841 | 0.0358771 | OSQP | OSQP | No measured crossover | 45 | 45 | 0 | 6.04738e-13 | 3.88578e-16 |
| QP-SPARSE-COUPLED | 10000 | 101.699 | 6.78198 | 14.9955 | 0.950062 | OSQP | cuOpt | 3000 | 45 | 45 | 0 | 1.84276e-08 | 1.82588e-11 |
