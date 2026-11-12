# Starbucks Calorie Classifier — Streamlit

A beginner-friendly Streamlit website that uses the supplied Starbucks dataset and
trained Decision Tree Classifier to predict whether a selected drink is in the
**Low Calorie** or **High Calorie** group.

## Project files

Keep these files together in one folder:

```text
starbucks_streamlit_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── starbucks.csv
└── starbucks_calories_classifier.pkl
```

The CSV and `.pkl` model are the two supplied project assets. The ZIP archive is
not needed after `starbucks.csv` has been extracted.

## Run on your computer

1. Install Python.
2. Open a terminal in this project folder.
3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

5. Streamlit will show a local URL. Open it in your browser.

## Upload to GitHub and deploy

1. Create a new repository on GitHub.
2. Upload all six files listed above.
3. On Streamlit Community Cloud, sign in with GitHub and create a new app.
4. Select your repository, branch, and `app.py` as the main file.
5. Deploy.

## How the prediction works

- The app reads `starbucks.csv`.
- It applies the same column cleaning and one-hot encoding approach used in the
  supplied notebook.
- It aligns the input columns with the trained model's saved feature names.
- The model returns `0` for Low Calorie or `1` for High Calorie.

The original notebook defines High Calorie as calories above the dataset median.
The calorie value shown on the page is the recorded value from the dataset; the
classifier predicts only the group.

## Important

Only use pickle files that you trust. The model file supplied with this project
is loaded by `app.py`. This is an educational demonstration, not medical or
dietary advice.
