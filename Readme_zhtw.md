# FastGS MLX
這是將 [FastGS](https://github.com/fastgs/FastGS) 重新以 MLX 與 Metal Code 實作。支援 `forward` 圖片輸出與 `backward` 模型訓練。

FastGS 與 3DGS 不同，3DGS 透過 `torch.Tensor.retain_grad()` 方式保留 `means2d` 相關資訊，以用於後訓練機制，然而 `retain_grad()` 在 MLX 上無對應實作；FastGS 使用 `viewspace_points` 保留 `means2d.grad` 與必要的資訊，以此讓 MLX 也可以透過 gradient 取得。

### Features
1. 使用 Cmake 建置專案
2. 建置 XCode 方便測試 Metal Code 與自動程式碼補齊功能。
3. 純 MLX 3DGS 訓練與顯示。
4. 目前 MLX C++ custom extension 實作非常少，此專案提供良好的模板可使用。

目前在剩餘進度為移植與測試 FastGS 後訓練機制(densification, clone and split)。

## 環境安裝

- 從 App Store 安裝 XCode
- 安裝 Cmake `brew install cmake`
- 安裝 Conda， Makefile 內容預設使用 Conda 環境

1. 建立 `fastgs_core` 虛擬環境，預設使用 Python 3.11 版本 
```shell
conda create -n fastgs_core python=3.11
```

2. 安裝需要的 pip package
安裝 pip package：
```shell
pip install mlx==0.30.0 nanobind cmake opencv-python plyfile
```

3. 安裝 fastgs_core:
```shell
pip install .
```

4. 安裝 spz：
```shell
git submodule update --init --recursive
cd submodules/spz
git checkout ef094fd1a96ca6ff414d72d7904ee4f4f6d97be9
pip install .
```

附註：
- mlx 版本並無限制必須為 `0.30.0`，唯 nanobind 有可能出現 `Incompatible function arguments`，如遇到此情況需要自行測試 mlx and nanobind.
- spz 在目前的最新版本中(b2a63b9204c2989de713e4e426a28eeaa415643e)，會有無法輸出的問題。

## 快速開始

## 資料集
目前只有支援 [3D Scanner iPhone App](https://3dscannerapp.com/) 掃描後資料訓練。
1. 使用具備 LiDAR 功能的 iPhone
2. 安裝 3D Scanner App
3. 以橫向方式(前置鏡頭在左側）拍攝
4. 選擇 Point Cloud 選項進行攝影
5. 輸出選擇「All Data」，以 AirDrop 傳輸到 Mac 上


## 執行指令
可參考 Makefile，預設使用的 Python 環境為 `CONDA_ENV ?= fastgs_core`，修改 `--data` 目錄以方便指向訓練資料集。

### 專案編譯與建置
- `env-check`: 確認環境
- `gen-primitive CLASS=Foo`：建立 mlx Primitive class Foo，並放置對應的目錄中。
- `pyext-build`：測試建置 python mlx extension
- `cmake-configure`, `test-build`, `test-run`：測試建置 XCode project，使用 `build/`
- `xcode-configure`, `xcode-build`:測試建置 XCode project，使用 `build_xcode/`
- `pip-install`, `pip-develop`, `pip-wheel`: 安裝 fastgs package


### FastGS 訓練與測試
#### FastGS Training
- `test-scanner`：將指定資料集輸出成 spz 與 side-by-side 圖片。
- `train-scanner-fixed`：以固定 gaussian 數量訓練。
- `train-scanner-fastgs`：正式的 FastGS 訓練
- `train-scanner-fastgs-smoke`：以較小的訓練次數測試 FastGS 訓練。
- `train-scanner-fastgs-bbox`：填充大量 gaussian 並訓練。
- `train-scanner-fastgs-densify`, `train-scanner-fastgs-densify2`, `train-scanner-fastgs-densify3`：使用不同參數訓練

目前推薦使用 `train-scanner-fastgs-densify3` 進行訓練。

#### Scripts

`scripts/` 提供一些可測試的 python scripts:
- `scripts/train_square.py`：使用固定圖片訓練 Gaussian，輸出 spz and side-by-side 圖片。
- `scripts/render_2048_cube_smoke.py`：輸出魔術方塊，輸出 spz。


## 致謝
- 3DGS
- FastGS

## 注意

- 本專案使用 Codex 5.3, 5.4 生成程式碼，因此會有許多奇妙的程式碼。Cuda code to Metal Code 是 Codex 不擅長的部分，容易生成簡化且錯誤的實作。Python 倒是沒有問題。
- MLX 不支援 boolean indices，可使用 range 替代。