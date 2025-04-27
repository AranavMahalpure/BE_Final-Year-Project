
# 🧠 Brain Tumor Detection and Segmentation using Deep Learning

This project aims to detect and segment brain tumors from MRI scans using deep learning techniques. It leverages the BRATS dataset and utilizes a U-Net architecture to perform pixel-level tumor segmentation. The application also includes tumor severity classification and is deployed via a user-friendly web interface.

## 📌 Features

- Brain tumor detection from MRI scans
- Tumor segmentation using U-Net architecture
- Tumor severity classification (e.g., Low-grade, High-grade)
- Web-based user interface for real-time image upload and diagnosis
- Dockerized deployment for easy scalability

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras, OpenCV
- **Web Frameworks:** Flask / Django (choose one), React.js
- **Data:** BRATS Dataset (Brain Tumor Segmentation)
- **Tools:** Docker, Git, WSL
- **Deployment:** Localhost or cloud (e.g., Render, Heroku, AWS)

## 🧬 Model Architecture

- **Preprocessing:** Resize, normalize, and convert images to grayscale if needed
- **Model:** U-Net (for segmentation), CNN (for classification)
- **Loss Function:** Dice coefficient + Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, IoU, Dice Score

## 📁 Project Structure

```plaintext
├── app.py                # Main Flask application
├── uploads/              # Folder to store uploaded MRI files
├── templates/
│   ├── index.html        # Upload page
│   └── result.html       # Prediction result page
├── tests/
│   └── test_app.py       # Pytest test cases
├── 3D_MRI_Brain_tumor_segmentation(35).h5  # Pre-trained Keras model
├── README.md             # This file
├── requirements.txt      # Python dependencies
## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AranavMahalpure/BE_Final-Year-Project.git
cd BE_Final-Year-Project
```

### 2. Set up and activate virtual environment

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
cd BE_Final-Year-Project
python app.py
```

### 4. Run the test cases

```bash
cd BE_Final-Year-Project
python -m pytest --maxfail=1 --disable-warnings -v
```
## 📊 Sample Output

- Input: Brain MRI Scan
- Output: Tumor Level Prediction

## 📚 Dataset

We use the [BRATS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/) for training and validation. Please ensure you download the dataset separately and place it in the `dataset/` folder.

## ✍️ Authors

- Aranav Mahalpure - Final Year Computer Engineering Student at I²IT College, Pune

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> ⚠️ This tool is for educational and research purposes only. It is not intended for clinical diagnosis.
