## Exploratory Data Analysis (EDA)

I put the visualization code in a separate file, eda.py, instead of
mixing it into data_processing.py. data_processing.py holds only
reusable cleaning logic that the future training script will depend on,
while eda.py is a standalone script anyone can run to regenerate the
exploratory plots (saved as PNGs in reports/) without needing a Jupyter
notebook.

I initially had eda.py call drop_unused_columns before generating the
plots, which broke the correlation heatmap and the demographics grid,
since those need columns (TotalCharges, gender, PhoneService,
MultipleLines) that drop_unused_columns removes. This surfaced an
ordering problem: dropping columns is a *conclusion* of the EDA, not a
step that should happen before it. eda.py now only loads and cleans the
data (load_data + clean_data), and explores the full dataset;
drop_unused_columns is meant to be used later, by the training script,
which needs the already-reduced dataset.

### Cleaning TotalCharges

TotalCharges is stored as text and has 11 blank values, all belonging
to customers with tenure = 0. Since a brand-new customer hasn't been
billed a total yet, I converted these blanks to 0 instead of dropping
the rows or leaving them null — the value has a clear logical
explanation, unlike a genuinely missing field.

### Numeric features: tenure, MonthlyCharges, TotalCharges

Boxplots of tenure and MonthlyCharges by Churn confirmed both are
strongly related to churn: customers who churn have a much lower median
tenure (~10 months vs ~38) and a higher median MonthlyCharges (~80 vs
~65).

A correlation heatmap showed tenure and TotalCharges are correlated at
0.83 — essentially redundant. I decided to drop TotalCharges and keep
tenure instead, since tenure is more interpretable on its own (months
as a customer) and isn't a mix of tenure and price the way TotalCharges
is. MonthlyCharges correlates more moderately with both (0.65 with
TotalCharges, 0.25 with tenure), so it stays as an independent feature.

### Categorical features

Contract, PaymentMethod, and InternetService all showed clear
differences in churn proportion: month-to-month contracts, electronic
check payment, and fiber optic internet are all associated with
noticeably higher churn than their alternatives.

For demographics and phone-related features, gender and PhoneService
showed almost no difference in churn between categories, and
MultipleLines showed only a weak one — all three were dropped.
SeniorCitizen, Partner, and Dependents did show real differences and
were kept.

The six internet-related service columns (OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies) all follow
a similar pattern by shape (since they share the same "No internet
service" category), but each one still describes a genuinely different
service the customer lacks. Unlike TotalCharges, there's no single
clear redundant pair to drop here, so all six were kept.

### Final column decisions

Dropped: customerID, TotalCharges, gender, PhoneService, MultipleLines.
Kept: everything else, including all six internet service columns.