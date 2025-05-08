
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
plt.plot(x, y, marker='X', linestyle='-', color='r', label="Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.legend()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'Name': ['Virat Kohli', 'Klassen', 'Rohit Sharma', 'Dhoni', 'Sam Curren'],
    'Total_Score': [8000, 1027, 6628, 5243, 1774]
})

# Create a bar plot
sns.barplot(x='Name', y='Total_Score', data=data, palette='Reds')

# Add title
plt.title("Seaborn Bar Plot")

# Show the plot
plt.show()
