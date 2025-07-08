import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import joblib

df = pd.read_excel('./personal loan-api/model training/Bank_Personal_Loan_Modelling.xlsx',sheet_name='Data')
print(df.head())
print(df.describe())

df=df[df['ZIP Code']>90000]

df.loc[df['Experience']<0,'Experience'] = 0 - df.loc[df['Experience']<0,'Experience']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))


below_50perc = df[df['Income']<=64]
sb.histplot(below_50perc["Income"],ax=axes[0,0],bins=100)
axes[0,0].title.set_text(f'{below_50perc["Income"].shape[0]} count of <64 anum members')
axes[1,0].pie(below_50perc["Personal Loan"].value_counts(),labels=below_50perc["Personal Loan"].value_counts().index,autopct="%1.1f%%")
axes[1,0].title.set_text('personal loan')

above_50perc_below150 = df[(df['Income']>64)&(df['Income']<=150)]
sb.histplot(above_50perc_below150['Income'],ax=axes[0,1],bins=100)
axes[0,1].title.set_text(f'{above_50perc_below150["Income"].shape[0]} count of >64 and <150 anum members')
axes[1,1].pie(above_50perc_below150["Personal Loan"].value_counts(),labels=above_50perc_below150["Personal Loan"].value_counts().index,autopct="%1.1f%%")
axes[1,1].title.set_text('personal loan')

sb.histplot(df.loc[df['Income']>150,'Income'],ax=axes[0,2],bins=50)
axes[0,2].title.set_text(f'{df.loc[df["Income"]>150,"Income"].shape[0]} count of >150 anum members')
axes[1,2].pie(df.loc[df["Income"]>150,"Personal Loan"].value_counts(),labels=df.loc[df["Income"]>150,"Personal Loan"].value_counts().index,autopct="%1.1f%%")
axes[1,2].title.set_text('personal loan')

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(20,10))
sb.barplot(df.Experience.value_counts(),ax=axes[0,0])
axes[0,0].title.set_text('Experience distribution')
for i in axes:
    for ax in i:
        ax.tick_params(axis='x',labelsize='8')

sb.scatterplot(data=df,x='Experience',y='Income',ax=axes[0,1])
axes[0,1].title.set_text('Exp VS income')

below_100inc = df[df['Income']<100]
print("\033[1m",round(below_100inc.shape[0]*100/df.shape[0],2),"% peoples are below 100$ anual income with average age of",round(below_100inc['Age'].mean(),2),"years\033[0m")
sb.barplot(below_100inc.Age.value_counts(),ax=axes[1,0])
axes[1,0].title.set_text('Below 100$ anual income age distribution')

above_100inc = df[df['Income']<100]
axes[1,1].pie(above_100inc.loc[(above_100inc["Age"]>=30)&(above_100inc["Age"]<=60),"Personal Loan"].value_counts(),labels=above_100inc.loc[(above_100inc["Age"]>=30)&(above_100inc["Age"]<=60),"Personal Loan"].value_counts().index,autopct="%1.1f%%")
axes[1,1].title.set_text('Personal loan below 100$')

df = df.dropna()

scale_cols = ['Income', 'CCAvg', 'Mortgage']

X = df.drop(columns='Personal Loan')
y = df['Personal Loan']

# The rest of the columns (left untouched)
pass_through_cols = [col for col in X.columns if col not in scale_cols]

# Preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), scale_cols),
        ('passthrough', 'passthrough', pass_through_cols)
    ]
)

# Full pipeline: Preprocessing + Model
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])

stdsc = StandardScaler()

model_df = df.dropna()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred))
disp.plot(cmap='Blues',values_format='d')
plt.show()

joblib.dump(pipeline,"./personal loan-api/model training/loan_model_stdsc_xgb_pipeline.pkl")

pipeline_loaded = joblib.load('./personal loan-api/model training/loan_model_stdsc_xgb_pipeline.pkl')
sampledf = df.sample(100)
prediction = pipeline_loaded.predict(sampledf.drop(columns='Personal Loan'))

y_test = sampledf['Personal Loan']
y_pred = prediction
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred))
disp.plot(cmap='Blues',values_format='d')
plt.show()