import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

# Gnerate some data
data = {
    'HoursStudied': [10,12,8,5,11,7,13,4,12,9],
    'ExamScore': [85,92,78, 60,88,70,96, 52, 90, 80]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Scatter plot
sns.scatterplot(x='HoursStudied', y='ExamScore', data=df)
plt.title('Hours Studied vs Exam Score')
plt.show()