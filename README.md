# Reusing the RVAT Reynolds number dependence experiment

This repo is a
[Calkit](https://github.com/calkit/calkit)
project that
reuses the RVAT Reynolds number dependence experimental results.
The goal here is to explore ideas to make reuse easier.

## Getting started

After installing [Calkit](https://github.com/calkit/calkit) and
setting your token in its config,
clone this project repo with:

```sh
calkit clone https://github.com/petebachant/reuse-rvat-re-dep
```

Then run the pipeline with:

```sh
calkit run
```

If you'd like to generate more figures,
data processing, etc.,
add stages to the pipeline in `dvc.yaml`.

If you'd like figures to be visible on [calkit.io](https://calkit.io),
add them to `calkit.yaml`.
