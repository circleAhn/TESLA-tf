# Real-time Calibration Model for Low-cost Sensor in Fine-grained Time Series

## Abstract

Precise measurements from sensors are crucial, but the data are usually collected from low-cost, low-tech systems, which are often inaccurate. Thus, they require further calibrations. To that end, we first identify three requirements for effective calibration under practical low-tech sensor conditions. Based on the requirements, we develop a model called TESLA, Transformer for effective sensor calibration utilizing logarithmic-binned attention. TESLA uses a high performance deep learning model, Transformers, to calibrate and capture non-linear components. At its core, it employs logarithmic binning, to minimize attention complexity. TESLA achieves fast and consistent real-time calibration, even with longer sequences and finer-grained time series in hardware-constrained systems. Experiments show that TESLA outperforms existing novel deep learning and newly crafted linear models in accuracy, calibration speed, and energy efficiency.

## Environment Requirements
- See `requirements.txt`

## File Directory Structure
- `dataset`: Contains experimental datasets
  - `./finedust_concentration_alldrop_firstcol_dataset/{region}_{sensor}_FirstCol.csv`
  - We will provide soon due to the space limitations

- `lbt`: Contains the experimental package
  - `./lbt`

- `result`: Stores TESLA weights and experimental results during training
  - `./{save_path}/{region}_{pm}_{window_size}.h5` - TESLA best weights for the validation dataset
  - `./{save_path}/{region}_{pm}_{window_size}.log` - Loss history during training
  - `./{save_path}/{region}_{pm}_{window_size}.png` - Plot for the test dataset using the best weights

## Command Examples
- **Training & Evaluation Process**  
  Evaluation scalar results are displayed in the console, and plots are saved to `{save_path}`.
  ```bash
  python main.py --save_path ./model --pm 10 --window_size 360 --gpu 0 --seed 2024
  ```

- **Only Evaluation Process**
  With model weights loaded from `{save_path}`.
  ```bash
  python main.py --save_path ./model --pm 10 --window_size 360 --gpu 0 --seed 2024
  ```
 
## Command-line Arguments
- `--save_path`
  Default: `"./model"`  
  Description: Directory path to save the model and results.

- `--pm`  
  Default: `10`
  Description: Select from `(1, 2.5, 10)`. Parameter for calibration.
  
- `--window_size`  
  Default: `360`  
  Description: Window size for data processing. Recommended values are `(15, 60, 360, 720, 1440)`.

- `--max_sensor_scale`
  Default: `300`  
  Description: Maximum sensor scale value.

- `--epoch`
  Default: `10`  
  Description: Number of training epochs.

- `--batch_size`
  Default: `32`  
  Description: Batch size for training.

- `--learning_rate`
  Default: `1e-4`  
  Description: Learning rate for the optimizer.

- `--shuffle_sensor_size`
  Default: `4`  
  Description: Number of shuffle sizes between sensors during training.

- `--gpu`
  Default: `None`
  Description: GPU device number for training. (e.g., `0`). If not specified, CPU will be used.

- `--seed`
  Default: `2024`
  Description: Random seed for reproducibility.


  
## Citation

If you would like to cite this paper, please make a copy of the following text. Thank you🤣

```bibtex
@article{ahn2025tesla,
      title={Real-Time Calibration Model for Low-Cost Sensor in Fine-Grained Time Series},
      author={Ahn, Seokho and Kim, Hyungjin and Shin, Sungbok and Seo, Young-Duk},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2025},
      month={Apr.},
      pages={3-11}
      volume={39},
      number={1},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/31974},
      DOI={10.1609/aaai.v39i1.31974},
}
```
