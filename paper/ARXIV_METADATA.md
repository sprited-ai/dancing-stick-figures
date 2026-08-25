# arXiv submission metadata

## Title

Dancing Stick Figures: A Synthetic Video Dataset, Renderer, and Diagnostic Evaluation Suite

## Authors

Jin Hyuk Cho (Sprited)

- ORCID: 0009-0001-1896-8242
- Email: jin@sprited.ai

## Abstract

Video generation is difficult to learn by doing. Real video datasets are large, the process that produced each frame is unknown, and a single quality score rarely identifies why a generated video failed. We introduce Dancing Stick Figures, a synthetic video dataset with a deterministic rendering pipeline designed for end-to-end experiments on one conventional GPU. It contains 1,430 six-second motions rendered from three cameras, for 4,290 videos and 514,800 labelled frames. Every frame remains linked to the joints, camera, body parameters, depth, surface normals, body-part labels, and source motion that produced it. This known state acts as an answer key for evaluation. We provide measurements of whether coloured limbs remain connected, whether the figure remains intact, and how its position and pose change over time. Controlled mistakes test which failures each measurement can and cannot detect. In a controlled 120-frame study, reversal leaves the reported time-symmetric motion signals unchanged and produces essentially the same FVD as the untouched real reference. The release includes a 0.85-GB 64x64 configuration, an optional 32x32 route, fixed data lists, evaluation scripts, checkpoints, and a Colab lesson that trains an image generator and then a video generator. Together, these components provide an accessible route for observing how video generation works and why it fails: a small, complete first experiment in video generation.

## Categories

- Primary: cs.CV (Computer Vision and Pattern Recognition)
- Cross-list: cs.LG (Machine Learning)

## Comments

7 pages, 4 figures, and 4 tables. Dataset, code, baseline checkpoints, and Colab notebook are linked from the paper.

## Links

- Dataset: https://huggingface.co/datasets/sprited/dancing-stick-figures
- Code: https://github.com/sprited-ai/dancing-stick-figures
- Models: https://huggingface.co/sprited/dancing-stick-figures-baselines
- Colab: https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab_v0_2.ipynb

## License at submission

Recommended: CC BY 4.0. Confirm this choice on the final arXiv submission screen because the license grant is irrevocable.

