---
title: Usage
description: Generate a simple sky map with PySM.
slug: 3.4.4/guides/usage
---

# Usage

Generate a full-sky map of the confirmed Galactic templates:

```python
import pysm
sky = pysm.Sky(nside=64, preset_strings=['d1', 's1', 'a1', 'f1'])
map = sky.get_emission(30.0)
```

The returned map is a tuple of HEALPix maps in RING ordering.
