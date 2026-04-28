README
Automotive Finance Portfolio Risk Notebook

Project files
- DSC_527_RS_Topic3_Risk_GenSynData_Part_1.ipynb
- financial_loan.csv
- automobile_loan_default_1.csv
- automobile_loan_default_2.csv

Project overview
This notebook builds an unified automotive finance dataset named auto_finance_combined_over_500k.csv by:
1. Loading one auto-loan source file and two vehicle-loan-default source files.
2. Standardizing column names and mapping source-specific fields into a shared schema.
3. Coercing key columns to numeric types and normalizing the default flag.
4. Harmonizing schemas across sources and filling missing values.
5. Creating engineered fields such as loan_to_income and monthly_payment_est.
6. Expanding the dataset with synthetic rows so the final dataset exceeds 500,000 records.
7. Producing report-support visuals and a small interactive Portfolio Risk Pulse Check quiz.

Python version
- Recommended: Python 3.11
- Google Colab is the easiest environment because the notebook already references google.colab and supports file upload and download there.

Libraries needed
Core libraries used directly in the notebook:
- numpy >= 2.3
- pandas == 2.2.2
- matplotlib == 3.9
- seaborn == 0.13.2
- plotly
- ipywidgets
- IPython

Colab-specific modules used in the notebook:
- google.colab.drive
- google.colab.files

Recommended install commands
If running in Google Colab:
- Most packages are already available.
- Run the first install cell in the notebook.
- If needed, also run:
  pip install plotly ipywidgets -q

If running locally in Jupyter Notebook or JupyterLab:
1. Create and activate a virtual environment.
2. Install the packages:
   pip install "numpy>=2.3" pandas==2.2.2 matplotlib==3.9 seaborn==0.13.2 plotly ipywidgets notebook
3. Start Jupyter:
   jupyter notebook

How to place the files
Option A: Google Colab
- Upload the notebook to Colab.
- Upload these CSV files into the same Colab session:
  - financial_loan.csv
  - automobile_loan_default_1.csv
  - automobile_loan_default_2.csv
- Keep the file names exactly as shown above.

Option B: Local Jupyter
- Place the notebook and the three CSV files in the same folder.
- Open that folder in Jupyter and run the notebook.

Important note about file names in Colab
If Colab renames an uploaded file to something like financial_loan (1).csv, either:
- rename the file in the Colab file pane back to financial_loan.csv, or
- update the file name variables in the notebook so they match the uploaded names.

How to execute the notebook
Run the notebook from top to bottom in order.

Suggested run order
1. Install libraries cell.
2. Import libraries cell.
3. Optional Google Drive mount cell.
4. Synthetic dataset generator helper cell.
5. Load, standardize, combine, and scale the data section.
6. Final dataset review cell.
7. Creative twist: Portfolio Risk Pulse Check section.
8. Summary statistics section.
9. Visualization cells.

What the notebook produces
Primary output file:
- auto_finance_combined_over_500k.csv

Intermediate and analytical outputs:
- summary_table with dataset metrics
- Distribution of Loan Amounts histogram
- Distribution of Borrower Income histogram
- Loan-to-Income Ratio Across Borrower Segments box plot
- Loan Amount vs Borrower Income by Default Risk scatter plot
- Correlation Heatmap of Financial and Risk Variables
- Portfolio Risk Pulse Check interactive quiz

Features and how to use them
1. Combined dataset creation
- The notebook reads the three source CSV files.
- It standardizes and maps fields to a shared structure so the sources can be concatenated.
- It then fills missing values and adds engineered variables.

2. Synthetic scaling
- The notebook expands the combined dataset to exceed 500,000 rows.
- Use this for performance testing, dashboard prototyping, and report visuals.

3. Portfolio Risk Pulse Check
- This is the creative twist in the notebook.
- It uses ipywidgets to ask simple borrower-risk questions.
- Best used in Google Colab or a notebook environment with widget support enabled.
- If widgets do not render, install ipywidgets and rerun that section.

4. Summary statistics
- Review the summary_table output for row count, default rate, median income, median loan amount, and median loan-to-income ratio.
- These values can be used directly in the technical report.

5. Visualizations
- Histograms show the distribution of loan amounts and borrower income.
- The box plot compares loan-to-income ratios across borrower segments.
- The scatter plot shows how income, loan amount, and default interact.
- The heatmap shows numeric relationships among key finance and risk variables.

Expected schema elements in the final dataset
Common fields used in the notebook include:
- income
- loan_amount
- int_rate or interest_rate depending on mapped output
- term
- default or default_flag depending on final normalized field name
- loan_to_income
- monthly_payment_est
- source
- record_origin

If a chart fails because of a column name mismatch, inspect the final dataframe columns and update the visualization cell to use the actual final field name.

How to save and download the final CSV
In Google Colab
1. Run all cells until the final dataset is created.
2. Confirm the file auto_finance_combined_over_500k.csv exists in the left-side Files pane.
3. Use the notebook's download cell if included.
4. Or click the three dots next to the CSV in the Files pane and choose Download.

In local Jupyter
1. Run all cells until the final dataset is created.
2. The CSV will be saved to the current working directory.
3. Open that folder in your file explorer and copy or move the file as needed.

Troubleshooting
1. DtypeWarning: mixed types
- This is common in financial CSV files.
- Fix by reading the CSV with low_memory=False.
- Then convert key numeric columns with pandas.to_numeric(errors="coerce").

2. ModuleNotFoundError: google.colab
- This occurs when running outside Colab.
- Comment out or skip google.colab import and drive/files helper cells.

3. ipywidgets does not display
- Install ipywidgets.
- Restart the kernel and rerun the notebook.
- In local JupyterLab, make sure widget support is enabled.

4. File not found
- Verify that the notebook and all three CSV files are in the same working folder, or update the file path variables.

5. Outliers distort the scatter plot
- Apply filtering or log scaling to income before plotting if needed for presentation clarity.

Recommended workflow for assignment submission
1. Run the notebook completely.
2. Save auto_finance_combined_over_500k.csv.
3. Capture the summary table and required charts.
4. Use the visualization comments and storyboard flow in the notebook to support the technical report.

Notes
- This notebook is intended for educational and portfolio use.
- Because the combined dataset includes synthetic expansion, analytical patterns should be interpreted carefully and described transparently in the technical report.
