# European Hotel Satisfaction Score Analysis

This repository contains the analysis of European hotel booking satisfaction scores. The goal of this project is to analyze customer satisfaction and provide actionable insights in order to improve hotel services.

## Project Structure

- `data/processed/Europe Hotel Booking Satisfaction Score.csv` – cleaned dataset used for analysis
- `notebooks/Hotel 2.ipynb` – exploratory analysis and visualizations
- `predict_satisfaction.py` – script to train a satisfaction prediction model
- `results/` – figures and summary document

## Usage

Install the required dependencies and run the prediction script:

```bash
pip install -r requirements.txt
python3 predict_satisfaction.py
```

The script will display the distribution of satisfaction levels and output model accuracy using a RandomForest classifier.
