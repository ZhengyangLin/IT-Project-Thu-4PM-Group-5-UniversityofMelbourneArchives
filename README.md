# University of Melbourne Archives

## Project Overview

This project is an architectural drawing recognition and metadata extraction tool for the University of Melbourne Archives. The system uses OCR to recognize text in drawing title blocks and extract fields such as Name of project, Name of drawing, Architect, Drawing number, Date, and Scale. It also supports manual review and result export to assist with archival organization.

The system uses local deployment. Users access the frontend through a browser, while the backend handles image recognition and result processing. Application startup has been verified on both a local Windows computer and an Ubuntu server. Access to the application on the Ubuntu server from other devices on the same local network has also been verified.

## Features

- Image recognition: Recognize architectural drawing title blocks and extract structured metadata.
- Task monitoring: View image counts, processing status, Folder statistics, and error messages, and filter tasks by folder or status.
- Batch processing and recheck: Process pending images and recheck individual images or multiple selected images.
- Manual review: Review recognition results against the original image, edit fields, and save the review results.
- Export: Export results for selected images or images within the current filters to an Excel file.
- Optional large language model assistance: Use the configured external large language model API to assist with field extraction and error correction. This feature can also be disabled.

## System Requirements

The following environment requirements are consistent with the installation instructions below.

| Item | Requirement |
| --- | --- |
| Operating system | Windows or Ubuntu |
| Python | 3.11, with an environment created using Conda |
| Node.js | 24.x, version 24.12.0 or later |
| Package managers | Conda, pip, and Yarn |
| GPU and CUDA | This guide uses an NVIDIA GPU and CUDA; select a compatible PaddlePaddle version based on the driver and supported CUDA version |
| Browser | Used to open the frontend page |
| Network | An internet connection is required to install dependencies, download models for the first time, and use an external large language model API |

# University of Melbourne Archives Installation and Startup Guide

For Windows and Ubuntu.

## 1. Download the Project

Open the [project download page](https://github.com/ZhengyangLin/IT-Project-Thu-4PM-Group-5-UniversityofMelbourneArchives), click Code → Download ZIP, and extract the downloaded archive.

## 2. Install Software

Open the links below, select software versions suitable for your operating system, and download and install them:

Conda: [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)

CUDA: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Node.js: [https://nodejs.org/en/download](https://nodejs.org/en/download) (choose a 24.x release, version 24.12.0 or later)

## 3. Enter the Project Folder

In the terminal, navigate to the project folder (the folder containing serve.py).

## 4. Backend Environment and Dependencies

In the current terminal, create the Conda environment:

```sh
conda create -n uma python=3.11 -y
```

Activate the Conda environment:

```sh
conda activate uma
```

Install the backend dependencies:

```sh
pip install -r requirements.txt
```

Check the NVIDIA GPU, driver version, and supported CUDA version:

```sh
nvidia-smi
```

Download and install PaddlePaddle: open the [PaddlePaddle installation page](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html). Based on the information shown above, select your operating system, Pip, GPU, and a compatible version.

Copy the installation command generated on the page and run it in the same terminal window.

## 5. Frontend Environment and Dependencies

Open a new terminal window and navigate to the front-end folder inside the project.

Download and install Yarn (Windows):

```sh
npm install -g yarn
```

Download and install Yarn (Ubuntu):

```sh
sudo npm install -g yarn
```

Install the frontend dependencies:

```sh
yarn install
```

## 6. Configure Settings and Add Images

Open config.yaml in the project folder, find llm, and fill in the following:

```yaml
llm:
  enabled: true
  base_url: "API endpoint"
  api_key_env: "API key"
  model: "Model name"
```

If you are not using a large language model, set enabled to false.

Place the images in the project folder using the following path:

```text
plan_sample/category_folder/NodeID/image.jpg
```

## 7. Start the Application

Use two terminal windows.

First terminal: navigate to the project folder.

Activate the Conda environment:

```sh
conda activate uma
```

Start the backend:

```sh
python serve.py
```

Second terminal: navigate to the front-end folder inside the project.

Start the frontend:

```sh
yarn dev --host 0.0.0.0 --port 5173 --strictPort
```

Keep both windows open. Open the address displayed in the frontend terminal in your browser.

The first run requires an internet connection to download the models.

## 8. Stop the Application

Press Ctrl+C in each terminal window.

Next time, you only need to follow Step 7.

## Usage Overview

1. Follow the installation and startup guide above to prepare the images and configuration, then start the frontend and backend. Open the address displayed in the frontend terminal in your browser.
2. View image tasks on the page and filter the images to be processed by folder or processing status.
3. Click `IS` to process pending images, or use `Check`, `Recheck`, and `Batch recheck` to process specific images.
4. For tasks with a Success status, click `Manual review` to check and edit the extraction results against the original image. Click `Save review` and confirm the save.
5. Click `Export` to export results. If images are selected, results for the selected images are exported. If no images are selected, results for all images within the current filters are exported, including those on other pages.

Detailed operating instructions are provided in a separate application user manual.

## License and Usage Restrictions

This software was developed by a student team at the University of Melbourne for the University of Melbourne Archives.

The project plans to make its source code publicly available to help other libraries, archives, and research institutions improve their efficiency.

The specific open-source license for this project has not yet been determined. Third-party components used in the project remain subject to their respective license terms.

### Third-Party Licenses

This project uses PaddlePaddle and PaddleOCR, both licensed under Apache License 2.0:

- [PaddlePaddle License](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE)
- [PaddleOCR License](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

Use of these third-party components is subject to their respective license terms. This section describes third-party licenses and does not replace this project's own license terms.

The applicable licenses for other third-party dependencies and separately downloaded pretrained models should be checked and followed individually.
