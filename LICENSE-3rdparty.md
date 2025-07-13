# Third-Party Licenses

This project includes code adapted from the following open-source projects:

## HiT-SR (Hierarchical Transformer for Efficient Image Super-Resolution)

- **Source**: https://github.com/XPixelGroup/HiT-SR
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) 2024 XPixel Group
- **Files**: `superscale/backends/native/hitsr/*`

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

## TSD-SR (One-Step Diffusion with Target Score Distillation)

- **Source**: https://github.com/Iceclear/TSD-SR
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) 2024 TSD-SR Authors
- **Files**: `superscale/backends/native/tsdsr/*`

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

## VARSR (Visual Autoregressive Modeling)

- **Source**: https://github.com/FoundationVision/VARSR
- **License**: MIT License
- **Copyright**: Copyright (c) 2024 Foundation Vision
- **Files**: `superscale/backends/native/varsr/*`

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## Additional Components

### Stable Diffusion 3

- **Source**: https://huggingface.co/stabilityai/stable-diffusion-3-medium
- **License**: Stability AI Community License Agreement
- **Note**: Model weights are downloaded separately and not included in this package. Users must agree to Stability AI's license terms when downloading these weights.

### BasicSR

- **Source**: https://github.com/XPixelGroup/BasicSR
- **License**: Apache License 2.0
- **Note**: HiT-SR is built on top of BasicSR framework

---

For the full text of each license, please refer to the LICENSE file in the respective model's directory within `superscale/backends/native/`.