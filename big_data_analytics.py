import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("BigDataAnalytics") \
    .getOrCreate()

# Sample data schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("salary", FloatType(), True),
    StructField("department", StringType(), True)
])

# Sample data 
data = [
    (1, "Alice", 34, 70000.00, "HR"),
    (2, "Bob", 45, 80000.00, "IT"),
    (3, "Charlie", 29, 60000.00, "IT"),
    (4, "David", 38, 50000.00, "Finance"),
    (5, "Edward", 50, 90000.00, "HR"),
    (6, "Fiona", 40, 40000.00, "Marketing"),
    (7, "George", 30, 30000.00, "Marketing"),
    (8, "Hannah", 33, 65000.00, "IT"),
]

# Create DataFrame
df = spark.createDataFrame(data, schema)

# Show the DataFrame
df.show()

# Data Summary
summary = df.describe()
summary.show()

# Data Cleaning
cleaned_df = df.dropna()

# Salary Statistics by Department
salary_stats = cleaned_df.groupBy("department").agg(
    F.avg("salary").alias("average_salary"),
    F.min("salary").alias("min_salary"),
    F.max("salary").alias("max_salary"),
    F.count("id").alias("employee_count")
)

salary_stats.show()

# Data Visualization
def plot_salary_distribution(dataframe):
    # Convert to Pandas for visualization
    pdf = dataframe.toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(pdf['salary'], bins=20, kde=True)
    plt.title('Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

plot_salary_distribution(cleaned_df)

# Scatter plot of Salary vs Age
def plot_salary_vs_age(dataframe):
    pdf = dataframe.toPandas()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='salary', data=pdf, hue='department', palette='deep')
    plt.title('Salary vs Age')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.grid()
    plt.legend()
    plt.show()

plot_salary_vs_age(cleaned_df)

# Correlation matrix
def plot_correlation_matrix(dataframe):
    pdf = dataframe.toPandas().corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(pdf, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

plot_correlation_matrix(cleaned_df)

# Machine Learning: Predicting Salary
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Feature Engineering
assembler = VectorAssembler(
    inputCols=["age"], 
    outputCol="features"
)

feature_df = assembler.transform(cleaned_df)

# Train-Test Split
train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=123)

# Initialize and Train Linear Regression Model
lr = LinearRegression(featuresCol='features', labelCol='salary')
lr_model = lr.fit(train_df)

# Model Summary
print(f"Coefficients: {lr_model.coefficients}, Intercept: {lr_model.intercept}")

# Predictions
predictions = lr_model.transform(test_df)
predictions.select("id", "age", "salary", "prediction").show()

# Evaluate the Model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Stop Spark Session
spark.stop()