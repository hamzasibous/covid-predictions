import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

def add_sentiment(df):
    df['sentiment'] = df['text'].apply(sentiment_analysis)
    return df

def train_model(df):
    X = df[['sentiment']]
    y = df['Cases']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return model

if __name__ == "__main__":
    df = pd.read_csv("matched_data.csv")
    df = add_sentiment(df)
    model = train_model(df)
    df['predicted_cases'] = model.predict(df[['sentiment']])
    df.to_csv("predictions.csv", index=False)