# Notices

GPU-Native-CFD is free software: you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version (`LICENSE`). It is distributed WITHOUT ANY WARRANTY.

## Code derived from other projects

| Files | Origin | Licence of the origin |
|---|---|---|
| `src/**` (algorithms and operation order), `tools/ref/**` (links against these codes) | OpenFOAM v2606, OpenCFD Ltd. (www.openfoam.com) | GPL-3.0-or-later |
| `src/solvers/gncHisa/**`, `src/ops/reconstruct.*`, parts of `src/fvm/` and `src/fields/` | HiSA (High Speed Aerodynamic solver), CSIR and contributors, gitlab.com/hisa/hisa, commit `271b4491` | GPL-3.0-or-later |
| `src/core/glibcPow.cuh`, `glibcPowData.cuh`, `glibcExp.cuh`, `glibcLog.cuh`, `glibcLogData.cuh` | GNU C Library 2.39 (`sysdeps/ieee754/dbl-64/e_pow.c`, `e_exp.c`, `e_log.c`, which derive from Arm Optimized Routines), reconstructed from the disassembly and data tables of Ubuntu's `libm.so.6` (libc6 2.39-0ubuntu8.9) | LGPL-2.1-or-later; used under GPL-3.0-or-later as permitted by section 3 of the LGPL-2.1 |
| `src/core/glibcTanh.cuh` | GNU C Library 2.39 (`s_tanh.c`, `s_expm1.c`), originally from fdlibm | LGPL-2.1-or-later (as above); fdlibm notice below |

The fdlibm notice carried by `s_tanh.c` and `s_expm1.c`:

> Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
>
> Developed at SunPro, a Sun Microsystems, Inc. business.
> Permission to use, copy, modify, and distribute this
> software is freely granted, provided that this notice
> is preserved.

Comments in the source cite the upstream files as "Atlas upstream/openfoam" and "upstream/hisa": these are the
OpenFOAM v2606 and HiSA sources as integrated in the authors' development repository.

## Data

`results/external/oneraM6-exp-schmitt-charpin/`: experimental pressure distributions on the ONERA M6 wing
(V. Schmitt and F. Charpin, AGARD AR-138, 1979), as distributed by the NPARC Alliance validation archive
(NASA Glenn Research Center, www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html); files unchanged.

The measurement records in `results/measured/` contain case dictionaries copied from the OpenFOAM tutorials and the
HiSA examples (GPL-3.0-or-later) and logs of OpenFOAM, HiSA, SPUMA and brae runs. SPUMA and brae themselves are not
part of this repository.

## Trademarks

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via
www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks. NVIDIA and CUDA are trademarks of NVIDIA
Corporation.
