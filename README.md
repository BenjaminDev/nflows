# An Exploration into Normalizing Flows
## What?
Normalizing Flows (NF) are a type of generative model. This means we can sample from the model. Let's say we want $I_{generated} \sim P_X(x)$ where $I$ is an image to be similar to:
```python
from pathlib import Path
from random import choice as P_X
from PIL import Image

data_dir = Path("~/pictures")
images = [Image.open(o) for o in data_dir.glob("*.png")]
I_real = P_X(images)
```
One way of evaluating a genrative model $P_X(x)$ is to look a the images $I_{real}$ and $I_{generated}$. Do they look like they came from the same "place". Human qualitative assessment.

This is a useful goal in it's own right as seen by the Internet going bossy for StableDiffusion[^stablediffusioncomment].
[^stablediffusioncomment]: Diffusion models share similarities to NFs but they are stochastic.

However, the goal in this work is to find a likelihood $P_{\theta}(X=x)$. This gives us the the probability of observing sample $X$ given $\theta$. $\theta$ is the "stuff" we looking for.

The idea! 

Following a common way to make generative models we can:
$$
P_{X}(x) = Z_{X}(z)|det
$$

 (Note: S)
 which can p
$$
P_X(x)
$$


 Consider the model of a dice roll:
```
def dice():
    from
```

download ocean data:
```
wget -nd -r -l 1 -A png https://data-dataref.ifremer.fr/stereo/AA_2014/2014-03-27_09-10-00_12Hz/input/cam1
```