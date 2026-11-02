# Data

`raw/rice_cammeo_osmancik.csv` is the **Rice (Cammeo and Osmancik)** dataset from the UCI Machine
Learning Repository, converted from the original ARFF file to CSV. Values and row order are
unchanged; only the header and file format differ.

- Source: <https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik>
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- 3,810 grains, 7 morphological features extracted from grain images, 2 classes
  (Cammeo 1,630 · Osmancik 2,180), no missing values.

Citation:

> Cinar, I. and Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence
> Methods. *International Journal of Intelligent Systems and Applications in Engineering*, 7(3),
> 188–194. <https://doi.org/10.18201/ijisae.2019355381>

SHA-256 of the CSV: `307084f7903e863f55737d9878c08cd6b017fceb0699801d1cca05439541c56f`
(checked by `tests/integration/test_real_data.py`).
